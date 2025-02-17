# Refer To: https://carla.readthedocs.io/en/latest/python_api/#carlaweatherparameters
import sys
import carla

SUN_PRESETS = {
    'day': (45.0, 0.0),
    'night': (-90.0, 0.0),
    'sunset': (0.5, 0.0)}

WEATHER_PRESETS = {
    'clear': [10.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0331, 0.0],
    'overcast': [80.0, 0.0, 0.0, 50.0, 2.0, 0.75, 0.1, 10.0, 0.0, 0.03, 0.0331, 0.0],
    'rain': [100.0, 80.0, 90.0, 100.0, 7.0, 0.75, 0.1, 100.0, 0.0, 0.03, 0.0331, 0.0]}

CAR_LIGHTS = {
    'None' : [carla.VehicleLightState.NONE],
    'Position' : [carla.VehicleLightState.Position],
    'LowBeam' : [carla.VehicleLightState.LowBeam],
    'HighBeam' : [carla.VehicleLightState.HighBeam],
    'Brake' : [carla.VehicleLightState.Brake],
    'RightBlinker' : [carla.VehicleLightState.RightBlinker],
    'LeftBlinker' : [carla.VehicleLightState.LeftBlinker],
    'Reverse' : [carla.VehicleLightState.Reverse],
    'Fog' : [carla.VehicleLightState.Fog],
    'Interior' : [carla.VehicleLightState.Interior],
    'Special1' : [carla.VehicleLightState.Special1],
    'Special2' : [carla.VehicleLightState.Special2],
    'All' : [carla.VehicleLightState.All]}

LIGHT_GROUP = {
    'None' : [carla.LightGroup.NONE],
    # 'Vehicle' : [carla.LightGroup.Vehicle],
    'Street' : [carla.LightGroup.Street],
    'Building' : [carla.LightGroup.Building],
    'Other' : [carla.LightGroup.Other]}

def apply_sun_presets(args, weather):
    """Uses sun presets to set the sun position"""
    if "sun" in args.__dict__:
        if args.sun in SUN_PRESETS:
            weather.sun_altitude_angle = SUN_PRESETS[args.sun][0]
            weather.sun_azimuth_angle = SUN_PRESETS[args.sun][1]
        else:
            print("[ERROR]: Command [--sun | -s] '" + args.sun + "' not known")
            sys.exit(1)


def apply_weather_presets(args, weather):
    """Uses weather presets to set the weather parameters"""
    if "weather" in args.__dict__:
        if args.weather in WEATHER_PRESETS:
            weather.cloudiness = WEATHER_PRESETS[args.weather][0]
            weather.precipitation = WEATHER_PRESETS[args.weather][1]
            weather.precipitation_deposits = WEATHER_PRESETS[args.weather][2]
            weather.wind_intensity = WEATHER_PRESETS[args.weather][3]
            weather.fog_density = WEATHER_PRESETS[args.weather][4]
            weather.fog_distance = WEATHER_PRESETS[args.weather][5]
            weather.fog_falloff = WEATHER_PRESETS[args.weather][6]
            weather.wetness = WEATHER_PRESETS[args.weather][7]
            weather.scattering_intensity = WEATHER_PRESETS[args.weather][8]
            weather.mie_scattering_scale = WEATHER_PRESETS[args.weather][9]
            weather.rayleigh_scattering_scale = WEATHER_PRESETS[args.weather][10]
            weather.dust_storm = WEATHER_PRESETS[args.weather][11]
        else:
            print("[ERROR]: Command [--weather | -w] '" + args.weather + "' not known")
            sys.exit(1)


def apply_weather_values(args, weather):
    """Set weather values individually"""
    if "azimuth" in args.__dict__:
        weather.sun_azimuth_angle = args.azimuth
    if "altitude" in args.__dict__:
        weather.sun_altitude_angle = args.altitude
    if "clouds" in args.__dict__:
        weather.cloudiness = args.clouds
    if "rain" in args.__dict__:
        weather.precipitation = args.rain
    if "puddles" in args.__dict__:
        weather.precipitation_deposits = args.puddles
    if "wind" in args.__dict__:
        weather.wind_intensity = args.wind
    if "fog" in args.__dict__:
        weather.fog_density = args.fog
    if "fogdist" in args.__dict__:
        weather.fog_distance = args.fogdist
    if "fogfalloff" in args.__dict__:
        weather.fog_falloff = args.fogfalloff
    if "wetness" in args.__dict__:
        weather.wetness = args.wetness
    if "scatteringintensity" in args.__dict__:
        weather.scattering_intensity = args.scatteringintensity
    if "miescatteringscale" in args.__dict__:
        weather.mie_scattering_scale = args.miescatteringscale
    if "rayleighscatteringscale" in args.__dict__:
        weather.rayleigh_scattering_scale = args.rayleighscatteringscale
    if "dust_storm" in args.__dict__:
        weather.dust_storm = args.dust_storm


def apply_lights_to_cars(args, world):
    if "cars" not in args.__dict__:
        return

    light_mask = carla.VehicleLightState.NONE
    for option in args.cars:
        light_mask |= CAR_LIGHTS[option][0]

    # Get all cars in level
    all_vehicles = world.get_actors()
    for ve in all_vehicles:
        if "vehicle." in ve.type_id:
            ve.set_light_state(carla.VehicleLightState(light_mask))

def apply_lights_manager(args, light_manager):
    if "lights" not in args.__dict__:
        return

    light_group = 'None'
    if "lightgroup" in args.__dict__:
        light_group = args.lightgroup

    # filter by group
    lights = light_manager.get_all_lights(LIGHT_GROUP[light_group][0]) # light_group


    i = 0
    while (i < len(args.lights)):
        option = args.lights[i]

        if option == "on":
            light_manager.turn_on(lights)
        elif option == "off":
            light_manager.turn_off(lights)
        elif option == "intensity":
            light_manager.set_intensity(lights, int(args.lights[i + 1]))
            i += 1
        elif option == "color":
            r = int(args.lights[i + 1])
            g = int(args.lights[i + 2])
            b = int(args.lights[i + 3])
            light_manager.set_color(lights, carla.Color(r, g, b))
            i += 3

        i += 1
