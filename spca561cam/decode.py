"""Frame decoding for the SPCA561: compressed-Bayer decoder and demosaicing.

The decompressor is a port of libv4lconvert/spca561-decompress.c
(C) 2005 Andrzej Szombierski, LGPL-2.1-or-later. It is JIT-compiled with
numba so that 352x288 frames decode in a few milliseconds.
"""
import numpy as np
import cv2
from numba import njit

from . import _tables as T

HEADER_LEN = 20  # bytes of frame header in front of the pixel data (rev 012a)

NBITS_A, TAB_A = T.NBITS_A, T.TAB_A
NBITS_B, TAB_B = T.NBITS_B, T.TAB_B
NBITS_C, TAB_C = T.NBITS_C, T.TAB_C
NBITS_D, TAB_D = T.NBITS_D, T.TAB_D
A_CURVE = T.A_CURVE
DIFF_ENCODING = T.DIFF_ENCODING
# The C table has 38 entries but is indexed with 19 + gkw for gkw up to 19.
ABS_CLAMP15 = np.append(T.ABS_CLAMP15, np.int32(15)).astype(np.int32)

FUN_A_TAB = np.array([12, 13, 14, 15, 16, 17, 18, 19, -12, -13, -14, -15,
                      -16, -17, -18, -19, -19], dtype=np.int32)
FUN_B_TAB1 = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
                      + [31] * 17 +
                      [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
                      dtype=np.int32)
FUN_B_TAB = np.array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                      -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16,
                      -17, -18, -19], dtype=np.int32)
FUN_C_TAB1 = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] + [23] * 21 +
                      [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22], dtype=np.int32)
FUN_C_TAB = np.array([8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, -9, -10, -11,
                      -12, -13, -14, -15, -16, -17, -18, -19], dtype=np.int32)
FUN_E_TABS = np.array([[0, -1, 1, -2, 2, -3, 3, -4],
                       [4, -5, 5, -6, 6, -7, 7, -8],
                       [8, -9, 9, -10, 10, -11, 11, -12],
                       [12, -13, 13, -14, 14, -15, 15, -16],
                       [16, -17, 17, -18, 18, -19, 19, -19]], dtype=np.int32)

BAD = 0xff


@njit(cache=True)
def _sext8(n):
    return n - 256 if n >= 128 else n


@njit(cache=True)
def _fun_f(cur_byte):
    """Returns (value, bits consumed)."""
    if cur_byte & 0x80:
        # 1xxxx s: 0,-1,1,-2,... for 0x80,0x88,0x90,...
        k = (cur_byte & 0x78) >> 3
        v = (k >> 1) if (k & 1) == 0 else -((k >> 1) + 1)
        return v, 5
    if (cur_byte & 0xc0) == 0x40:
        k = (cur_byte & 0x3c) >> 2
        v = 8 + (k >> 1) if (k & 1) == 0 else -(9 + (k >> 1))
        return v, 6
    if (cur_byte & 0xe0) == 0x20:
        k = (cur_byte & 0x1e) >> 1
        if k == 7:  # 0x2e is not a valid code
            return BAD, 0
        v = 16 + (k >> 1) if (k & 1) == 0 else -(17 + (k >> 1))
        return v, 7
    return BAD, 0


@njit(cache=True)
def _decode(width, height, inbuf, tmp,
            nbits_a, tab_a, nbits_b, tab_b, nbits_c, tab_c, nbits_d, tab_d,
            a_curve, abs_clamp15, diff_enc,
            fa_tab, fb_tab1, fb_tab, fc_tab1, fc_tab, fe_tabs):
    accum = np.zeros(512, dtype=np.int32)
    i_hits = np.zeros(512, dtype=np.int32)

    xwidth = width + 6
    off_up_right = 2 - 2 * xwidth
    off_up_left = -2 - 2 * xwidth
    pixel_u = 0
    saved_pixel_ur = 0
    pixel_x = 0
    pixel_y = 2

    for i in range(width):
        tmp[xwidth * 2 + 3 + i] = inbuf[HEADER_LEN + i]
        tmp[xwidth * 3 + 3 + i] = inbuf[HEADER_LEN + width + i]

    ip = HEADER_LEN + width * 2
    op = xwidth * 4 + 3
    bucket = 0
    bitfill = 0

    for block in range((height - 2) * width // 32):
        if bitfill < 8:
            bucket = ((bucket << 8) | inbuf[ip]) & 0xffffffff
            ip += 1
            bitfill += 8
        cur_byte = (bucket >> (bitfill & 7)) & 0xff

        var_7 = 0
        if (cur_byte & 0x80) == 0:
            var_7 = 0
            bitfill -= 1
        elif (cur_byte & 0xc0) == 0x80:
            var_7 = 1
            bitfill -= 2
        else:
            var_7 = 2
            bitfill -= 2

        for b_it in range(32):
            if bitfill < 8:
                bucket = ((bucket << 8) | inbuf[ip]) & 0xffffffff
                ip += 1
                bitfill += 8
            cur_byte = (bucket >> (bitfill & 7)) & 0xff

            pixel_l = np.int32(tmp[op - 2])
            pixel_ur = np.int32(tmp[op + off_up_right])
            pixel_ul = np.int32(tmp[op + off_up_left])

            d_l = diff_enc[0x100 + pixel_ul - pixel_l]
            d_c = diff_enc[0x100 + pixel_u - pixel_ul]
            d_r = diff_enc[0x100 + pixel_ur - pixel_u]

            if pixel_x < 2:
                pixel_u = np.int32(tmp[op - xwidth * 2])
                pixel_l = pixel_u
                pixel_ul = pixel_u
                pixel_ur = np.int32(tmp[op + off_up_right])
                d_l = 0
                d_c = 0
                d_r = diff_enc[0x100 + pixel_ur - pixel_u]
            elif pixel_x > width - 3:
                d_r = 0

            multiplier = 4
            index = d_r + d_c * 8 + d_l * 64

            if (pixel_l + pixel_u * 2 <= 144 and (pixel_y & 1) == 0
                    and (b_it & 3) == 0 and d_r < 5 and d_c < 5 and d_l < 5):
                multiplier = 1
            elif pixel_l <= 48 and d_l <= 4 and d_c <= 4 and d_l >= 1 and d_c >= 1:
                multiplier = 2
            elif var_7 == 1:
                multiplier = 2
            elif d_c + d_l >= 11 or var_7 == 2:
                multiplier = 8

            hits = i_hits[index]
            acc = accum[index]
            if hits < 7:
                bitfill -= nbits_a[cur_byte]
                gkw = tab_a[cur_byte]
                if gkw == 0xfe:  # fun_A
                    bucket = ((bucket << 8) | inbuf[ip]) & 0xffffffff
                    ip += 1
                    bitfill -= 4
                    gkw = fa_tab[(bucket >> (bitfill & 0xff)) & 0xf]
                    if bitfill < 8:
                        bucket = ((bucket << 8) | inbuf[ip]) & 0xffffffff
                        ip += 1
                        bitfill += 8
            elif hits >= acc:
                bitfill -= nbits_b[cur_byte]
                gkw = tab_b[cur_byte]
                if cur_byte == 0:  # fun_B
                    bucket = ((bucket << 8) | inbuf[ip]) & 0xffffffff
                    ip += 1
                    bitfill -= 7
                    t = ((bucket >> (bitfill & 0xff)) & 0x7f) - 68
                    if bitfill < 8:
                        bucket = ((bucket << 8) | inbuf[ip]) & 0xffffffff
                        ip += 1
                        bitfill += 8
                    if t < 0 or t > 47:
                        gkw = BAD
                    else:
                        gkw = fb_tab[fb_tab1[t]]
            elif hits * 2 >= acc:
                bitfill -= nbits_c[cur_byte]
                gkw = tab_c[cur_byte]
                if cur_byte < 2:  # fun_C
                    if gkw == 0xfe:
                        bucket = ((bucket << 8) | inbuf[ip]) & 0xffffffff
                        ip += 1
                        bitfill -= 1
                        gkw = 7 if ((bucket >> (bitfill & 0xff)) & 1) == 0 else -8
                    elif gkw != 0xff:
                        gkw = BAD
                    else:
                        bucket = ((bucket << 8) | inbuf[ip]) & 0xffffffff
                        ip += 1
                        bitfill -= 7
                        t = ((bucket >> (bitfill & 0xff)) & 0x7f) - 72
                        if t < 0 or t > 43:
                            gkw = BAD
                        else:
                            if bitfill < 8:
                                bucket = ((bucket << 8) | inbuf[ip]) & 0xffffffff
                                ip += 1
                                bitfill += 8
                            gkw = fc_tab[fc_tab1[t]]
            elif hits * 4 >= acc:
                bitfill -= nbits_d[cur_byte]
                gkw = tab_d[cur_byte]
                if cur_byte < 4:  # fun_D
                    if gkw == 0xfd or gkw == 0xfc or gkw == 0xfe or gkw == 0xff:
                        n = 1 if gkw >= 0xfc and gkw <= 0xfd else (2 if gkw == 0xfe else 3)
                        bucket = ((bucket << 8) | inbuf[ip]) & 0xffffffff
                        ip += 1
                        bitfill -= n
                        v = (bucket >> (bitfill & 0xff)) & ((1 << n) - 1)
                        if gkw == 0xfd:
                            gkw = 12 if v == 0 else -13
                        elif gkw == 0xfc:
                            gkw = 13 if v == 0 else -14
                        elif gkw == 0xfe:
                            if v == 0:
                                gkw = 14
                            elif v == 1:
                                gkw = -15
                            elif v == 2:
                                gkw = 15
                            else:
                                gkw = -16
                        else:
                            if v == 4:
                                gkw = 16
                            elif v == 5:
                                gkw = -17
                            elif v == 6:
                                gkw = 17
                            elif v == 7:
                                gkw = -18
                            elif v == 2:
                                bitfill -= 1
                                gkw = -19 if ((bucket >> (bitfill & 0xff)) & 1) else 18
                            elif v == 3:
                                bitfill -= 1
                                gkw = 18
                            else:
                                gkw = BAD
            elif hits * 8 >= acc:  # fun_E
                if (cur_byte & 0xf0) >= 0x80:
                    bitfill -= 4
                    gkw = fe_tabs[0, (cur_byte >> 4) & 7]
                elif (cur_byte & 0xc0) == 0x40:
                    bitfill -= 5
                    gkw = fe_tabs[1, (cur_byte >> 3) & 7]
                elif (cur_byte & 0xe0) == 0x20:
                    bitfill -= 6
                    gkw = fe_tabs[2, (cur_byte >> 2) & 7]
                elif (cur_byte & 0xf0) == 0x10:
                    bitfill -= 7
                    gkw = fe_tabs[3, (cur_byte >> 1) & 7]
                elif (cur_byte & 0xf8) == 8:
                    bitfill -= 8
                    gkw = fe_tabs[4, cur_byte & 7]
                else:
                    gkw = BAD
            else:
                gkw, used = _fun_f(cur_byte)
                bitfill -= used

            if gkw == BAD:
                return -3

            tmp1 = (pixel_u + pixel_l) * 3 - pixel_ul * 2
            if tmp1 < 0:
                tmp1 += 3
            tmp2 = a_curve[19 + gkw] * multiplier
            if tmp2 < 0:
                tmp2 += 1
            val = (tmp1 >> 2) - (tmp2 >> 1)
            tmp[op] = 0 if val < 0 else (255 if val > 255 else val)
            op += 1

            pixel_u = saved_pixel_ur
            saved_pixel_ur = pixel_ur

            pixel_x += 1
            if pixel_x == width:
                op += 6
                pixel_x = 0
                pixel_y += 1

            accum[index] += abs_clamp15[19 + gkw]
            if i_hits[index] == 15:
                i_hits[index] = 8
                accum[index] //= 2
            else:
                i_hits[index] += 1
    return 0


def decode_spca561(frame, width, height):
    """Decode one compressed SPCA561 frame into a (height, width) Bayer image.

    Returns None if the bitstream is corrupt (e.g. a dropped USB packet).
    """
    # Padding: the bit reader may run a few bytes past the payload.
    buf = np.zeros(len(frame) + 64, dtype=np.uint8)
    buf[:len(frame)] = np.frombuffer(frame, dtype=np.uint8)
    if len(frame) < HEADER_LEN + 2 * width:
        return None
    tmp = np.zeros((height + 2) * (width + 6), dtype=np.uint8)
    ret = _decode(width, height, buf, tmp,
                  NBITS_A, TAB_A, NBITS_B, TAB_B, NBITS_C, TAB_C, NBITS_D, TAB_D,
                  A_CURVE, ABS_CLAMP15, DIFF_ENCODING,
                  FUN_A_TAB, FUN_B_TAB1, FUN_B_TAB, FUN_C_TAB1, FUN_C_TAB, FUN_E_TABS)
    if ret != 0:
        return None
    return tmp.reshape(height + 2, width + 6)[2:, 3:3 + width].copy()


# The sensor delivers GBRG Bayer (V4L2_PIX_FMT_SGBRG8). OpenCV names Bayer
# conversions after the second row, so GBRG maps to COLOR_BayerGR2BGR.
BAYER_TO_BGR = cv2.COLOR_BayerGR2BGR


def debayer(bayer):
    return cv2.cvtColor(bayer, BAYER_TO_BGR)
