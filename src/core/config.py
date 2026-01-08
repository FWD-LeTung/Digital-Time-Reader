import yaml
import os

class Config:
    def __init__(self, config_path="configs/settings.yaml"):
        if not os.path.exists(config_path):
            config_path = os.path.join(os.getcwd(), config_path)
            
        if not os.path.exists(config_path):
            print(f"Warning: Config file not found at {config_path}")
            self._config = {}
        else:
            with open(config_path, 'r') as f:
                self._config = yaml.safe_load(f)

    @property
    def pose_model_path(self):
        return self._config.get('model', {}).get('pose_path', 'models/clock_pose_v1.pt')

    @property
    def pose_conf(self):
        return self._config.get('model', {}).get('pose_conf', 0.5)

    @property
    def digit_model_path(self):
        return self._config.get('model', {}).get('digit_path', 'models/digit_rec_v1.pt')
    
    @property
    def digit_conf(self):
        return self._config.get('model', {}).get('digit_conf', 0.5)

    @property
    def camera_id(self):
        return self._config.get('camera', {}).get('id', 0)
    
    @property
    def camera_width(self):
        return self._config.get('camera', {}).get('width', 640)

    @property
    def camera_height(self):
        return self._config.get('camera', {}).get('height', 480)

    @property
    def warp_size(self):
        w = self._config.get('processing', {}).get('warp_width', 320)
        h = self._config.get('processing', {}).get('warp_height', 128)
        return (w, h)

    @property
    def debug_mode(self):
        return self._config.get('app', {}).get('debug_mode', True)

# Initialize global object for use everywhere
settings = Config()