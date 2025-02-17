# python CarRain/seqToVideo.py --data_root data/seqTownsCombineDay/test --types background rainy
python CarRain/seqToVideo.py --data_root data/seqTownsCombineDay/test --types semantic_segmentation instance_segmentation 
python CarRain/seqToVideo.py --data_root data/seqTownsCombineSunset/test --types background rainy \
    semantic_segmentation instance_segmentation 
python CarRain/seqToVideo.py --data_root data/seqTownsCombineNight/test --types background rainy \
    semantic_segmentation instance_segmentation 