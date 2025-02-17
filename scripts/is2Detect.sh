# scenes=(ClearDay/seqTown01Clear \
#     ClearDay/seqTown02Clear \
#     ClearDay/seqTown03Clear \
#     ClearDay/seqTown04Clear \
#     ClearDay/seqTown05Clear \
#     ClearDay/seqTown06Clear \
#     ClearDay/seqTown07Clear \
#     ClearSunset/seqTown07ClearSunset \
#     ClearDay/seqTown10Clear \
#     ClearSunset/seqTown10ClearSunset)

# for scene in ${scenes[@]}
# do
#     echo python CarRain/isToDetect.py -c configs/$scene.yaml
# done

# python CarRain/isToDetect.py -c configs/ClearDay/seqTown01Clear.yaml
# python CarRain/isToDetect.py -c configs/ClearDay/seqTown02Clear.yaml
# python CarRain/isToDetect.py -c configs/ClearDay/seqTown03Clear.yaml
# python CarRain/isToDetect.py -c configs/ClearDay/seqTown04Clear.yaml
# python CarRain/isToDetect.py -c configs/ClearDay/seqTown05Clear.yaml
# python CarRain/isToDetect.py -c configs/ClearDay/seqTown06Clear.yaml
python CarRain/isToDetect.py -c configs/ClearNight/seqTown06ClearNight.yaml
python CarRain/isToDetect.py -c configs/ClearSunset/seqTown06ClearSunset.yaml
# python CarRain/isToDetect.py -c configs/ClearDay/seqTown07Clear.yaml
# python CarRain/isToDetect.py -c configs/ClearSunset/seqTown07ClearSunset.yaml
# python CarRain/isToDetect.py -c configs/ClearDay/seqTown10Clear.yaml
# python CarRain/isToDetect.py -c configs/ClearSunset/seqTown10ClearSunset.yaml