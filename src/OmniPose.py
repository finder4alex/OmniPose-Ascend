# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""OmniPose network"""
from src.models import get_omnipose
from src.config import config

if config["MODELARTS"]["IS_MODEL_ARTS"]:
    pretrained = config["MODELARTS"]["CACHE_INPUT"] + config["MODEL"]["PRETRAINED"]
else:
    pretrained = config["TRAIN"]["CKPT_PATH"] + config["MODEL"]["PRETRAINED"]


def create_model():
    return get_omnipose(config, True)
