import math
import argparse
import numpy as np
import cv2
import e2p
import f2p

# 角度の単位変換
def DEG2RAD(a):
    return a * np.pi / 180.0

# メイン関数
def main():
    #
    # 引数の設定
    #
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Convert a 360 degree image to a perspective image.')

    # 必須パラメータ

    # 入力画像ファイル名
    parser.add_argument(
        '-i',
        required=True,
        help='Input image')
    # 出力画像ファイル名    
    parser.add_argument(
        '-o',
        required=True,
        help='Output image')
    # 横方向の画角    
    parser.add_argument(
        '--fov_u',
        required=True,
        type=float,
        help='Horizontal field of view')
    # 縦方向の画角    
    parser.add_argument(
        '--fov_v',
        required=True,
        type=float,
        help='Vertical field of view')

    # オプションパラメータ

    # 入力画像の種類
    parser.add_argument(
        '--image-type',
        type=int,
        default=0,
        help='Input image type; 0: Equirectanguler, 1: Fisheye')
    # 画像の内挿方法
    parser.add_argument(
        '--interp-type',
        type=int,
        default=1,
        help='Interpolation type; 0: Nearest, 1: Linear, 2: Cubic')
    # 出力画像の横サイズ
    parser.add_argument(
        '--ow',
        type=int,
        default=0,
        help='Horizontal image size')
    # 出力画像の縦サイズ
    parser.add_argument(
        '--oh',
        type=int,
        default=0,
        help='Vertical image size')
    # 横方向の回転角度
    parser.add_argument(
        '--ua',
        type=float,
        default=0,
        help='Horizontal viewing angle')
    parser.add_argument(
        '--va',
        type=float,
        default=0,
        help='Vertical viewing angle')
    # 画像面の回転角度
    parser.add_argument(
        '--za',
        type=float,
        default=0,
        help='Image rotation angle')
    # スケーリングパラメータ
    parser.add_argument(
        '-s',
        type=float,
        default=1,
        help='Scaling parameter')


    # 引数解析
    args = parser.parse_args()

    # 入力画像の読み込み
    src_img = cv2.imread(args.i)

    # パラメータ計算
    src_h, src_w = src_img.shape[: 2]
    f = src_h / np.pi    
    if args.ow <= 0:
        fov_u_param = 2.0 * np.tan(DEG2RAD(args.fov_u) / 2.0)
        args.ow = int(fov_u_param * f)
    if args.oh <= 0:
        fov_v_param = 2.0 * np.tan(DEG2RAD(args.fov_v) / 2.0)
        args.oh = int(fov_v_param * f)

    # 透視投影画像への変換
    
    if args.image_type == 0:
        converter = e2p.E2P(src_w,
                            src_h,
                            args.ow,
                            args.oh,
                            args.s,
                            args.interp_type)
    else:
        converter = f2p.F2P(src_w,
                            args.ow,
                            args.oh,
                            args.s,
                            args.interp_type)
        
    converter.generate_map(DEG2RAD(args.ua),
                           DEG2RAD(args.va),
                           DEG2RAD(args.za))
    dst_img = converter.generate_image(src_img)
        
    # 画像の保存
    cv2.imwrite(args.o, dst_img)
    
if __name__ == '__main__':
    main()
