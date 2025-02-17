import carla
import os, platform
import random
import queue
if platform.system() == "Windows":
    from pynput import keyboard
import signal

from config import Config
from utils import printDivider,create_dir_not_exist,rename_dir_existed

from environment import apply_sun_presets,apply_weather_presets,apply_weather_values,apply_lights_to_cars,apply_lights_manager

class CARLASimulator:
    def __init__(self):
        pass
    
    def init(self,args):
        printDivider()
        print("Init CARLASimulator...")
        self.args = args
        if platform.system() != "Windows":
            self.args.useKeyboard = False
        self.seqRoot = os.path.join(args.dataRoot,args.name)
        if os.path.exists(self.seqRoot):
            rename_dir_existed(self.seqRoot)
        create_dir_not_exist(self.seqRoot)

        # Connect to the client and retrieve the world object
        self.client = carla.Client(Config.serverIP, Config.serverPort)
        self.client.set_timeout(20.0)
        self.maps = self.client.get_available_maps()
        self.world = self.client.load_world(args.map)
        self.client.reload_world()

        # synchronous mode
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1 / args.fps
        self.world.apply_settings(settings)

        # Init world
        self.initWorld()

        # Init environment
        self.initEnvironment(self.world,args)

        # Init walkers
        self.initWalkers(args)

        # Create cameras
        assert len(args.cameras) == len(args.imgTypes), "cameras count must equal to imgTypes count"
        self.cameras = []
        for cameraType,imgType in zip(args.cameras,args.imgTypes):
            self.initCamera(cameraType,imgType)
        printDivider()

        # Create Keyboard Listener
        if self.args.useKeyboard:
            self.listener = keyboard.Listener(
                on_press=self.on_press,
                on_release=self.on_release) 
            
        # Status
        self.isRunning = False
        self.frame = 0

    def initWorld(self):
        print("Init World...")
        # Get the blueprint library and filter for the vehicle blueprints
        vehicle_blueprints = self.world.get_blueprint_library().filter('*vehicle*')

        # Get the map's spawn points
        spawn_points = self.world.get_map().get_spawn_points()

        # Spawn vehicles randomly distributed throughout the map 
        # for each spawn point, we choose a random vehicle from the blueprint library
        for _ in range(0,self.args.vehicleCount):
            vehicle = self.world.try_spawn_actor(random.choice(vehicle_blueprints), random.choice(spawn_points))
            if vehicle:
                vehicle.set_autopilot(True)


        # Spawn our car
        main_blueprint = random.choice(vehicle_blueprints)
        if "main_vehicle" in self.args:
            for blueprint in vehicle_blueprints:
                if blueprint.id == self.args.main_vehicle:
                    main_blueprint = blueprint
                    break

        myCar = self.world.try_spawn_actor(main_blueprint, random.choice(spawn_points))
        while myCar == None:
            myCar = self.world.try_spawn_actor(main_blueprint, random.choice(spawn_points))
        myCar.set_autopilot(True)
        self.myCar = myCar

        # Get init position of our car
        bound_x = 0.5 + myCar.bounding_box.extent.x
        bound_y = 0.5 + myCar.bounding_box.extent.y
        bound_z = 0.5 + myCar.bounding_box.extent.z

        # Create a transform to place the camera on top of the vehicle
        initPos = self.args.cameraPos
        self.cameraInitTrans = carla.Transform(
                                    carla.Location( x= initPos[0][0] * bound_x + initPos[0][1],
                                                    y=  initPos[1][0] * bound_y + initPos[1][1],
                                                    z=  initPos[2][0] * bound_z + initPos[2][1]
                                                    ),
                                    carla.Rotation(pitch=5) )

    def initWalkers(self, args):
        if "walkerCount" not in args:
            return
        print("Init Walkers...")
        # Spawn Walkers
        n_walkers = args.walkerCount
        percentagePedestriansCrossing = args.percentWalkerCrossing     # how many pedestrians will walk through the road
        
        walker_bps = self.world.get_blueprint_library().filter("walker.*.*")
        SpawnActor = carla.command.SpawnActor
        walkers_list = []
        all_id = []
        
        if n_walkers > 0:
            # 1. take all the random locations to spawn
            spawn_points = []
            for i in range(n_walkers):
                spawn_point = carla.Transform()
                loc = self.world.get_random_location_from_navigation()
                if (loc != None):
                    spawn_point.location = loc
                    spawn_points.append(spawn_point)
            # 2. we spawn the walker object
            batch = []
            walker_speed = []
            for spawn_point in spawn_points:
                walker_bp = random.choice(walker_bps)
                # set as not invincible
                if walker_bp.has_attribute('is_invincible'):
                    walker_bp.set_attribute('is_invincible', 'false')
                # set the max speed
                if walker_bp.has_attribute('speed'):
                    # walking
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[1])
                else:
                    print("Walker has no speed")
                    walker_speed.append(0.0)
                batch.append(SpawnActor(walker_bp, spawn_point))
            results = self.client.apply_batch_sync(batch, True)
            walker_speed2 = []
            for i in range(len(results)):
                if results[i].error:
                    print(results[i].error)
                else:
                    walkers_list.append({"id": results[i].actor_id})
                    walker_speed2.append(walker_speed[i])
            walker_speed = walker_speed2
            # 3. we spawn the walker controller
            batch = []
            walker_controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
            for i in range(len(walkers_list)):
                batch.append(SpawnActor(walker_controller_bp, carla.Transform(), walkers_list[i]["id"]))
            results = self.client.apply_batch_sync(batch, True)
            for i in range(len(results)):
                if results[i].error:
                    print(results[i].error)
                else:
                    walkers_list[i]["con"] = results[i].actor_id
            # 4. we put altogether the walkers and controllers id to get the objects from their id
            for i in range(len(walkers_list)):
                all_id.append(walkers_list[i]["con"])
                all_id.append(walkers_list[i]["id"])
            
        all_actors = self.world.get_actors(all_id)

        # ensures client has received the last transform of the walkers we have just created
        self.world.tick()

        # 5. initialize each controller and set target to walk to (list is [controler, actor, controller, actor ...])
        # set how many pedestrians can cross the road
        self.world.set_pedestrians_cross_factor(percentagePedestriansCrossing)
        for i in range(0, len(all_id), 2):
            # start walker
            all_actors[i].start()
            # set walk to random point
            all_actors[i].go_to_location(self.world.get_random_location_from_navigation())
            # max speed
            all_actors[i].set_max_speed(float(walker_speed[int(i/2)]))
        
        print('Spawned %d walkers.' % (len(walkers_list)))

    def initEnvironment(self, world, args):
        print("Init Environment...")
        weather = world.get_weather()

        class Environment:
            def __init__(self):
                pass
        
        environment = Environment()
        for key in args.environment:
            environment.__dict__[key] = args.environment[key]

        # apply presets
        apply_sun_presets(environment, weather)
        apply_weather_presets(environment, weather)

        # apply weather values individually
        apply_weather_values(environment, weather)

        world.set_weather(weather)

        # apply car light changes
        apply_lights_to_cars(environment, world)

        apply_lights_manager(environment, world.get_lightmanager())

    def initCamera(self, cameraType,imgType):
        assert cameraType in Config.cameraPaths, "Camera {} is not support!".format(cameraType)
        print("Creating Camera {}...".format(cameraType))
        create_dir_not_exist(os.path.join(self.seqRoot,Config.cameraPaths[cameraType]))

        if cameraType == "rgb":
            camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        elif cameraType == "depth":
            camera_bp = self.world.get_blueprint_library().find('sensor.camera.depth')
        elif cameraType == "is":
            camera_bp = self.world.get_blueprint_library().find('sensor.camera.instance_segmentation')
        elif cameraType == "ss":
            camera_bp = self.world.get_blueprint_library().find('sensor.camera.semantic_segmentation')

        camera_bp.set_attribute('image_size_x', str(self.args.resolution[0]))
        camera_bp.set_attribute('image_size_y', str(self.args.resolution[1]))
        camera = self.world.spawn_actor(camera_bp, self.cameraInitTrans, attach_to=self.myCar)

        # Image Queue
        image_queue = queue.Queue()

        # Start camera
        camera.listen(image_queue.put)

        self.cameras.append({
            "cameraType": cameraType,
            "camera": camera,
            "queue": image_queue,
            "outPath": os.path.join(self.seqRoot,Config.cameraPaths[cameraType]),
            "imgType": imgType
        })

    # Save image to disk
    def saveImage(self,image, cameraType, outPath, imgType):
        filePath = os.path.join(outPath,'%06d' % image.frame +"."+imgType )
        if cameraType == "rgb":
            image.save_to_disk(filePath)
        elif cameraType == "depth":
            image.save_to_disk(filePath,carla.ColorConverter.LogarithmicDepth)
        elif cameraType == "is":
            image.save_to_disk(filePath)
        elif cameraType == "ss":
            image.save_to_disk(filePath,carla.ColorConverter.CityScapesPalette)
        print("Save to ",filePath)


    def on_press(self,key):
        pass

    def on_release(self,key):
        if key == keyboard.Key.esc:
            self.on_exit()

    # Exit
    def on_exit(self):
        print("Exit Simulation!")
        self.isRunning = False
        try:
            self.myCar.destroy()
            for item in self.cameras:
                item["camera"].destroy()
        except:
            print("Destroy Error!")
        print("Destroy Successfully!")

    # Run simulator
    def run(self):
        print("Run Simulator")
        self.isRunning = True
        if self.args.useKeyboard:
            self.listener.start()
        signal.signal(signal.SIGINT,onExit)


        while self.isRunning:
            print("Frame ",self.frame)
            self.world.tick()
            for item in self.cameras:
                if not self.isRunning:
                    break
                cameraType = item["cameraType"]
                imgType = item["imgType"]
                image_queue = item["queue"]
                outPath = item["outPath"]
                try:
                    image = image_queue.get(timeout=Config.imgQueueTimeout)
                    self.saveImage(image,cameraType,outPath,imgType)
                except:
                    break
            self.frame += 1
            if self.frame >= self.args.frameCount:
                self.on_exit()
                break

simulator = CARLASimulator()

def onExit(signum, frame):
    simulator.on_exit()