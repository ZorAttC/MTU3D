import modules.third_party.mask3d.resunet as resunet
import modules.third_party.mask3d.res16unet as res16unet
from modules.third_party.mask3d.res16unet import (
    Res16UNet34C,
    Res16UNet34A,
    Res16UNet14A,
    Res16UNet34D,
    Res16UNet18D,
    Res16UNet18B,
    Custom30M,
)
from modules.third_party.mask3d.memory_unet import (
    Res16UNet34C_MM,
    Res16UNet34C_MM_FF,
)
# from Swin3D.models import Swin3DUNet
# from modules.third_party.swin3d.Swin3D.models import Swin3DUNet