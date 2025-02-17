import carla
import queue
import random
import os,sys
import shutil

from pynput import keyboard

def on_press(key):
    pass


def on_release(key):
    if key == keyboard.Key.esc:
        print("Quit")
        ego_vehicle.destroy()
        camera.destroy()
        sys.exit()


if __name__=="__main__":
    # Connect to the client and retrieve the world object
    client = carla.Client('localhost', 2000)
    maps = client.get_available_maps()
    print("Maps:",maps)
    world = client.get_world()

    # synchronous mode
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.1
    world.apply_settings(settings)

    # Retrieve the spectator object
    spectator = world.get_spectator()

    # Get the location and rotation of the spectator through its transform
    transform = spectator.get_transform()

    location = transform.location
    rotation = transform.rotation
    print("Location:",location)
    print("Rotation:",rotation)

    # Set the spectator with an empty transform
    # spectator.set_transform(carla.Transform())

    # Get the blueprint library and filter for the vehicle blueprints
    vehicle_blueprints = world.get_blueprint_library().filter('*vehicle*')
    # print("Vehicles:",vehicle_blueprints)

    # Get the map's spawn points
    spawn_points = world.get_map().get_spawn_points()
    # print("Spawn Points:",spawn_points)

    # Spawn 50 vehicles randomly distributed throughout the map 
    # for each spawn point, we choose a random vehicle from the blueprint library
    for i in range(0,50):
        world.try_spawn_actor(random.choice(vehicle_blueprints), random.choice(spawn_points))

    ego_vehicle = world.spawn_actor(random.choice(vehicle_blueprints), random.choice(spawn_points))
    ego_vehicle.set_autopilot(True)
    bound_x = 0.5 + ego_vehicle.bounding_box.extent.x
    bound_y = 0.5 + ego_vehicle.bounding_box.extent.y
    bound_z = 0.5 + ego_vehicle.bounding_box.extent.z
    print("EGO:",ego_vehicle)
    # Create a transform to place the camera on top of the vehicle
    camera_init_trans = carla.Transform(
                    carla.Location(x= -(bound_x + 1.0), z=bound_z+0.15),
                    carla.Rotation(pitch=5))

    # We create the camera through a blueprint that defines its properties
    camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '1920')
    camera_bp.set_attribute('image_size_y', '1080')
    # camera_bp.set_attribute('fov', '110')

    # We spawn the camera and attach it to our ego vehicle
    camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=ego_vehicle)

    # Image Queue
    image_queue = queue.Queue()

    # Start camera
    camera.listen(image_queue.put)
    # camera.listen(lambda image: image.save_to_disk('out\\%06d.png' % image.frame))

    for vehicle in world.get_actors().filter('*vehicle*'):
        vehicle.set_autopilot(True)

    if os.path.exists("out"):
        shutil.rmtree("out")


    listener = keyboard.Listener(
            on_press=on_press,
            on_release=on_release) 
    listener.start()

    print("Start Loop")

    while True:
        world.tick()
        image = image_queue.get()
        image.save_to_disk('out\\%06d.jpg' % image.frame)
        print("Frame ",image.frame)
