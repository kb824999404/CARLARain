import os

with open("imglist.txt","r") as f:
    imgList = f.readlines()

for item in imgList:
    name = item.strip().split(".")[0]
    print('<tr>')
    print('<td style="padding: 0;width=20%;"><img src="Docs/CARLARainDataset/background/{}..jpg" /></td>'.format(name))
    print('<td style="padding: 0;width=20%;"><img src="Docs/CARLARainDataset/rainy/{}.jpg" /></td>'.format(name))
    print('<td style="padding: 0;width=20%;"><img src="Docs/CARLARainDataset/depth/{}.png" /></td>'.format(name))
    print('<td style="padding: 0;width=20%;"><img src="Docs/CARLARainDataset/semantic_segmentation/{}.png" /></td>'.format(name))
    print('<td style="padding: 0;width=20%;"><img src="Docs/CARLARainDataset/instance_segmentation/{}.png" /></td>'.format(name))
    print('</tr>')
