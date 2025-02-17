# scenes=(ClearDay/seqTown01Clear \
#     ClearDay/seqTown02Clear \
#     ClearDay/seqTown03Clear \
#     ClearDay/seqTown04Clear \
#     ClearDay/seqTown05Clear \
#     ClearDay/seqTown06Clear \
#     ClearNight/seqTown06ClearNight \
#     ClearSunset/seqTown06ClearSunset \
#     ClearDay/seqTown07Clear \
#     ClearSunset/seqTown07ClearSunset \
#     ClearDay/seqTown10Clear \
#     ClearSunset/seqTown10ClearSunset)
# for scene in ${scenes[@]}
# do
#     echo python CarRain/main.py -c configs/$scene.yaml
# done

# python CarRain/main.py -c configs/ClearDay/seqTown01Clear.yaml
# python CarRain/main.py -c configs/ClearDay/seqTown02Clear.yaml
# python CarRain/main.py -c configs/ClearDay/seqTown03Clear.yaml
# python CarRain/main.py -c configs/ClearDay/seqTown04Clear.yaml
# python CarRain/main.py -c configs/ClearDay/seqTown05Clear.yaml
# python CarRain/main.py -c configs/ClearDay/seqTown06Clear.yaml
# python CarRain/main.py -c configs/ClearNight/seqTown06ClearNight.yaml
# python CarRain/main.py -c configs/ClearSunset/seqTown06ClearSunset.yaml
# python CarRain/main.py -c configs/ClearDay/seqTown07Clear.yaml
# python CarRain/main.py -c configs/ClearSunset/seqTown07ClearSunset.yaml
# python CarRain/main.py -c configs/ClearDay/seqTown10Clear.yaml
# python CarRain/main.py -c configs/ClearSunset/seqTown10ClearSunset.yaml


python CarRain/main.py -c configs/RainDay/seqTown01Rain.yaml
python CarRain/main.py -c configs/RainDay/seqTown02Rain.yaml
python CarRain/main.py -c configs/RainDay/seqTown03Rain.yaml
python CarRain/main.py -c configs/RainDay/seqTown04Rain.yaml
python CarRain/main.py -c configs/RainDay/seqTown05Rain.yaml
python CarRain/main.py -c configs/RainDay/seqTown06Rain.yaml
python CarRain/main.py -c configs/RainDay/seqTown07Rain.yaml
python CarRain/main.py -c configs/RainDay/seqTown10Rain.yaml
