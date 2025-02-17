# scenes=(ClearDay/seqTown01Clear \
#     ClearDay/seqTown02Clear \
#     ClearDay/seqTown03Clear \
#     ClearDay/seqTown04Clear \
#     ClearDay/seqTown05Clear \
#     ClearDay/seqTown06Clear \
#     ClearDay/seqTown07Clear \
#     ClearDay/seqTown10Clear \
#     ClearSunset/seqTown01ClearSunset \
#     ClearSunset/seqTown02ClearSunset \
#     ClearSunset/seqTown03ClearSunset \
#     ClearSunset/seqTown04ClearSunset \
#     ClearSunset/seqTown05ClearSunset \
#     ClearSunset/seqTown06ClearSunset \
#     ClearSunset/seqTown07ClearSunset \
#     ClearSunset/seqTown10ClearSunset \
#     ClearNight/seqTown01ClearNight \
#     ClearNight/seqTown02ClearNight \
#     ClearNight/seqTown03ClearNight \
#     ClearNight/seqTown04ClearNight \
#     ClearNight/seqTown05ClearNight \
#     ClearNight/seqTown06ClearNight \
#     ClearNight/seqTown07ClearNight \
#     ClearNight/seqTown10ClearNight)

# for scene in ${scenes[@]}
# do
#     echo python CarRain/ssToCityscapes.py -c configs/$scene.yaml
# done

python CarRain/ssToCityscapes.py -c configs/ClearDay/seqTown01Clear.yaml
python CarRain/ssToCityscapes.py -c configs/ClearDay/seqTown02Clear.yaml
python CarRain/ssToCityscapes.py -c configs/ClearDay/seqTown03Clear.yaml
python CarRain/ssToCityscapes.py -c configs/ClearDay/seqTown04Clear.yaml
python CarRain/ssToCityscapes.py -c configs/ClearDay/seqTown05Clear.yaml
python CarRain/ssToCityscapes.py -c configs/ClearDay/seqTown06Clear.yaml
python CarRain/ssToCityscapes.py -c configs/ClearDay/seqTown07Clear.yaml
python CarRain/ssToCityscapes.py -c configs/ClearDay/seqTown10Clear.yaml
python CarRain/ssToCityscapes.py -c configs/ClearSunset/seqTown01ClearSunset.yaml
python CarRain/ssToCityscapes.py -c configs/ClearSunset/seqTown02ClearSunset.yaml
python CarRain/ssToCityscapes.py -c configs/ClearSunset/seqTown03ClearSunset.yaml
python CarRain/ssToCityscapes.py -c configs/ClearSunset/seqTown04ClearSunset.yaml
python CarRain/ssToCityscapes.py -c configs/ClearSunset/seqTown05ClearSunset.yaml
python CarRain/ssToCityscapes.py -c configs/ClearSunset/seqTown06ClearSunset.yaml
python CarRain/ssToCityscapes.py -c configs/ClearSunset/seqTown07ClearSunset.yaml
python CarRain/ssToCityscapes.py -c configs/ClearSunset/seqTown10ClearSunset.yaml
python CarRain/ssToCityscapes.py -c configs/ClearNight/seqTown01ClearNight.yaml
python CarRain/ssToCityscapes.py -c configs/ClearNight/seqTown02ClearNight.yaml
python CarRain/ssToCityscapes.py -c configs/ClearNight/seqTown03ClearNight.yaml
python CarRain/ssToCityscapes.py -c configs/ClearNight/seqTown04ClearNight.yaml
python CarRain/ssToCityscapes.py -c configs/ClearNight/seqTown05ClearNight.yaml
python CarRain/ssToCityscapes.py -c configs/ClearNight/seqTown06ClearNight.yaml
python CarRain/ssToCityscapes.py -c configs/ClearNight/seqTown07ClearNight.yaml
python CarRain/ssToCityscapes.py -c configs/ClearNight/seqTown10ClearNight.yaml