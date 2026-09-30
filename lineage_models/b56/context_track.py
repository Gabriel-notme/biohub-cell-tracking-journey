"""Seven-frame residual 3D scene encoder followed by ten-step track attention."""
from track_video import TrackVideoNet
from cell_event import CellEventNet
class ContextTrackNet(TrackVideoNet):
    patch_shape=(12,24,24)
    def __init__(self,base_config,**kwargs):
        super().__init__(**kwargs)
        self.frame=CellEventNet(**base_config)
        del self.frame_proj
    def encode_frame(self,x):return self.frame.encode(x.float())
