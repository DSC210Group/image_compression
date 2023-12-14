import os
import torch
import cv2
from dct import LinearDCT, dct_2d, idct_2d, apply_linear_2d


class Compressor:
    def __init__(self, in_features=8, compress_rate=1):
        self.block_size = in_features
        # self.linear_dct = LinearDCT(in_features, 'dct')
        # self.linear_idct = LinearDCT(in_features, 'idct')
        self.luminance_matrix = torch.asarray(
            [[16, 11, 10, 16, 24, 40, 51, 61],
             [12, 12, 14, 19, 26, 58, 60, 55],
             [14, 13, 16, 24, 40, 57, 69, 56],
             [14, 17, 22, 29, 51, 87, 80, 62],
             [18, 22, 37, 56, 68, 109, 103, 77],
             [24, 35, 55, 64, 81, 104, 113, 92],
             [49, 64, 78, 87, 103, 121, 120, 101],
             [72, 92, 95, 98, 112, 100, 103, 99]]
        ) // compress_rate
        self.chrominance_matrix = torch.asarray(
            [[17, 18, 24, 47, 99, 99, 99, 99],
             [18, 21, 26, 66, 99, 99, 99, 99],
             [24, 26, 56, 99, 99, 99, 99, 99],
             [47, 66, 99, 99, 99, 99, 99, 99],
             [99, 99, 99, 99, 99, 99, 99, 99],
             [99, 99, 99, 99, 99, 99, 99, 99],
             [99, 99, 99, 99, 99, 99, 99, 99],
             [99, 99, 99, 99, 99, 99, 99, 99]]
        ) // compress_rate

    def compress(self, img):
        rows = (img.shape[0] - 1) // self.block_size + 1
        cols = (img.shape[1] - 1) // self.block_size + 1
        patched_img = torch.zeros((rows * self.block_size, cols * self.block_size, 3))
        patched_img[:img.shape[0], :img.shape[1], :] = torch.from_numpy(img).float()
        patched_img = patched_img.permute(2, 0, 1)
        patched_img = self.chroma(patched_img)
        patched_img = self.transform(patched_img)
        patched_img = self.quantize(patched_img)

        return patched_img, img.shape

    def chroma(self, img):
        """input [3:rows:cols]"""
        chroma_img = torch.empty(img.shape)
        chroma_img[0, :, :] = 0.299 * img[0, :, :] + 0.587 * img[1, :, :] + 0.114 * img[2, :, :]
        chroma_img[1, :, :] = img[2, :, :] - chroma_img[0, :, :]
        chroma_img[2, :, :] = img[0, :, :] - chroma_img[0, :, :]
        return chroma_img

    def inverse_chroma(self, chroma_img):
        """input [3:rows:cols] YUV"""
        img = torch.empty(chroma_img.shape)
        img[2, :, :] = chroma_img[1, :, :] + chroma_img[0, :, :]
        img[0, :, :] = chroma_img[2, :, :] + chroma_img[0, :, :]
        img[1, :, :] = (chroma_img[0, :, :] - 0.299 * img[0, :, :] - 0.114 * img[2, :, :]) / 0.587
        return img

    def transform(self, img):
        """input [3:rows:cols]"""
        # img -= 128
        # rows = img.shape[1] // self.block_size
        # cols = img.shape[2] // self.block_size
        # for i in range(rows):
        #     for j in range(cols):
        #         img[:, i * self.block_size:(i + 1) * self.block_size,
        #         j * self.block_size:(j + 1) * self.block_size] = apply_linear_2d(
        #             img[:, i * self.block_size:(i + 1) * self.block_size,
        #             j * self.block_size:(j + 1) * self.block_size], self.linear_dct)
        return img

    def inverse_transform(self, img):
        """input [3:rows:cols]"""
        # rows = img.shape[1] // self.block_size
        # cols = img.shape[2] // self.block_size
        # for i in range(rows):
        #     for j in range(cols):
        #         img[:, i * self.block_size:(i + 1) * self.block_size,
        #         j * self.block_size:(j + 1) * self.block_size] = apply_linear_2d(
        #             img[:, i * self.block_size:(i + 1) * self.block_size,
        #             j * self.block_size:(j + 1) * self.block_size], self.linear_idct)
        # img += 128
        return img

    def quantize(self, img):
        """input [3:rows:cols] YUV"""
        rows = img.shape[1] // self.block_size
        cols = img.shape[2] // self.block_size
        for i in range(rows):
            for j in range(cols):
                img[0, i * self.block_size:(i + 1) * self.block_size,
                j * self.block_size:(j + 1) * self.block_size] = torch.round(
                    img[0, i * self.block_size:(i + 1) * self.block_size,
                    j * self.block_size:(j + 1) * self.block_size] / self.luminance_matrix)
                img[1:, i * self.block_size:(i + 1) * self.block_size,
                j * self.block_size:(j + 1) * self.block_size] = torch.round(
                    img[1:, i * self.block_size:(i + 1) * self.block_size,
                    j * self.block_size:(j + 1) * self.block_size] / self.chrominance_matrix)
        return img

    def dequantize(self, img):
        """input [3:rows:cols]"""
        rows = img.shape[1] // self.block_size
        cols = img.shape[2] // self.block_size
        for i in range(rows):
            for j in range(cols):
                img[0, i * self.block_size:(i + 1) * self.block_size,
                j * self.block_size:(j + 1) * self.block_size] = \
                    img[0, i * self.block_size:(i + 1) * self.block_size,
                    j * self.block_size:(j + 1) * self.block_size] * self.luminance_matrix
                img[1:, i * self.block_size:(i + 1) * self.block_size,
                j * self.block_size:(j + 1) * self.block_size] = \
                    img[1:, i * self.block_size:(i + 1) * self.block_size,
                    j * self.block_size:(j + 1) * self.block_size] * self.chrominance_matrix
        return img

    def decompress(self, img, org_size: tuple):
        img = self.dequantize(img)
        img = self.inverse_transform(img)
        img = self.inverse_chroma(img)
        unpatched_img = img.permute(1, 2, 0)[:org_size[0], :org_size[1], :]
        return unpatched_img.to(dtype=torch.uint8).numpy()

    def approximate_size(self, img):
        """input [3:rows:cols]"""

        def _sub_size(_sub_img):
            """input [3:8:8]"""
            _count = 0
            for _i in range(3):
                _x = 7
                _y = 7
                while True:
                    # print((_x, _y))
                    if _sub_img[_i, _x, _y]:
                        _count += 1
                        _x -= 1
                        _y += 1
                        if _y == 8 or _x == -1:
                            _y = _x + _y - 1 - 7
                            _x = 7
                            if _y < 0:
                                _x = _x + _y
                                _y = 0
                                if _x < 0:
                                    break
                    else:
                        break
            return 8 * 8 * 3 - _count

        size = 0
        rows = img.shape[1] // self.block_size
        cols = img.shape[2] // self.block_size
        for i in range(rows):
            for j in range(cols):
                size += _sub_size(img[:, i * self.block_size:(i + 1) * self.block_size,
                                  j * self.block_size:(j + 1) * self.block_size] == 0) + 6
        return size


class DCTCompressor(Compressor):
    def __init__(self, in_features=8, compress_rate=1):
        super().__init__(in_features, compress_rate)
        self.linear_dct = LinearDCT(in_features, 'dct')
        self.linear_idct = LinearDCT(in_features, 'idct')

    def transform(self, img):
        """input [3:rows:cols]"""
        img -= 128
        rows = img.shape[1] // self.block_size
        cols = img.shape[2] // self.block_size
        for i in range(rows):
            for j in range(cols):
                img[:, i * self.block_size:(i + 1) * self.block_size,
                j * self.block_size:(j + 1) * self.block_size] = apply_linear_2d(
                    img[:, i * self.block_size:(i + 1) * self.block_size,
                    j * self.block_size:(j + 1) * self.block_size], self.linear_dct)
        return img

    def inverse_transform(self, img):
        """input [3:rows:cols]"""
        rows = img.shape[1] // self.block_size
        cols = img.shape[2] // self.block_size
        for i in range(rows):
            for j in range(cols):
                img[:, i * self.block_size:(i + 1) * self.block_size,
                j * self.block_size:(j + 1) * self.block_size] = apply_linear_2d(
                    img[:, i * self.block_size:(i + 1) * self.block_size,
                    j * self.block_size:(j + 1) * self.block_size], self.linear_idct)
        img += 128
        return img


class WHTCompressor(Compressor):
    def __init__(self, in_features=8, compress_rate=1):
        super().__init__(in_features, compress_rate)
        self.hadamard_matrix = torch.asarray([
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, -1, 1, -1, 1, -1, 1, -1],
            [1, 1, -1, -1, 1, 1, -1, -1],
            [1, -1, -1, 1, 1, -1, -1, 1],
            [1, 1, 1, 1, -1, -1, -1, -1],
            [1, -1, 1, -1, -1, 1, -1, 1],
            [1, 1, -1, -1, -1, -1, 1, 1],
            [1, -1, -1, 1, -1, 1, 1, -1]
        ]).float()

    # def quantize(self, img):
    #     return img
    #
    # def dequantize(self, img):
    #     return img

    def transform(self, img):
        """input [3:rows:cols]"""
        img -= 128
        rows = img.shape[1] // self.block_size
        cols = img.shape[2] // self.block_size
        for i in range(rows):
            for j in range(cols):
                img[:, i * self.block_size:(i + 1) * self.block_size,
                j * self.block_size:(j + 1) * self.block_size] = \
                    self.hadamard_matrix @ img[:, i * self.block_size:(i + 1) * self.block_size,
                                           j * self.block_size:(j + 1) * self.block_size] @ self.hadamard_matrix / 8
        return img

    def inverse_transform(self, img):
        """input [3:rows:cols]"""
        rows = img.shape[1] // self.block_size
        cols = img.shape[2] // self.block_size
        for i in range(rows):
            for j in range(cols):
                img[:, i * self.block_size:(i + 1) * self.block_size,
                j * self.block_size:(j + 1) * self.block_size] = \
                    self.hadamard_matrix @ img[:, i * self.block_size:(i + 1) * self.block_size,
                                           j * self.block_size:(j + 1) * self.block_size] @ self.hadamard_matrix / 8
        img += 128
        return img


def try_dct(img):
    x = torch.from_numpy(img).permute(2, 0, 1).float()
    linear_dct = LinearDCT(8, 'dct')
    res = apply_linear_2d(x, linear_dct)
    error = torch.abs(dct_2d(x) - res)
    assert error.max() < 1e-1, (error, error.max())
    linear_idct = LinearDCT(8, 'idct')
    ires = apply_linear_2d(res, linear_idct)
    error = torch.abs(idct_2d(res) - ires)
    assert error.max() < 1e-1, (error, error.max())
    error = torch.abs(x - ires)
    assert error.max() < 1e-1, (error, error.max())
    cv2.imshow("original", img)
    cv2.imshow("final", ires.to(dtype=torch.uint8).permute(1, 2, 0).numpy())


def try_wht():
    compressor = WHTCompressor(8, 1)
    img = cv2.imread('./data/kodak/kodim01.png')
    print(img.shape)  # Print image shape
    cv2.imshow("original", img)
    result, org_size = compressor.compress(img)
    # print(result)
    print(result[:, 256:264, 256:264])
    final = compressor.decompress(result, org_size)
    # Display cropped image
    cv2.imshow("final", final)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    compressors = {'DCT': DCTCompressor(), 'WHT': WHTCompressor()}
    output = dict()

    for filename in os.listdir('./data/kodak'):
        img = cv2.imread('./data/kodak/' + filename)
        # print(img.shape)  # Print image shape
        # try_dct(img[280:288, 150:158])
        # cv2.imshow("original", img)
        for compressor_name, compressor in compressors.items():
            result, org_size = compressor.compress(img)
            approx_size = compressor.approximate_size(result)
            output[compressor_name + '_' + filename] = \
                f'{compressor_name}_{filename} approx. size: {approx_size}Bytes, bitrate: {approx_size * 8 / img.shape[0] / img.shape[1]}'
            # print(result)
            final = compressor.decompress(result, org_size)
            cv2.imwrite('./data/kodak_result/' + compressor_name + '/' + filename, final)
            print(filename, ' done.')
        # Display cropped image
        # cv2.imshow("final", final)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    for filename in sorted(output.keys()):
        print(output[filename])


if __name__ == '__main__':
    main()
    # try_wht()
