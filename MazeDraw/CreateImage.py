from PIL import Image, ImageDraw

class CreateImage():
    def __init__(self):
        pass

if __name__ == '__main__':
    # (0, 0, 0, 0. 0)～(0, 1, 1, 1, 0)までの15種類と(1, 1, 1, 1, 1)
    # 上記を二進数としてファイル名を決める

    for i in range(15):
        left = (i & 8) >> 3
        down = (i & 4) >> 2
        right = (i & 2) >> 1
        up = (i & 1)

        #
        im = Image.new('RGB',(30,30),(255,255,255))
        draw = ImageDraw.Draw(im)
        draw.rectangle((0,0,29,29), outline=(230,230,230))
        if left == 1:
            draw.rectangle((0,0,1,29), fill=(0,0,0))
        if down == 1:
            draw.rectangle((0,28,29,29), fill=(0,0,0))
        if right == 1:
            draw.rectangle((28,0,29,29), fill=(0,0,0))
        if up == 1:
            draw.rectangle((0,0,29,1), fill=(0,0,0))

        im.save('mazetile{0}.jpg'.format(i), quality=95)

    im = Image.new('RGB',(30,30), (0,128,0))
    im.save('mazetile15.jpg', quality=95)