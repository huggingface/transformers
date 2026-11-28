# Copyright 2026 Google DeepMind and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Channel layouts shared by WeatherNext 2 configuration and feature extraction."""


class WeatherNext2ChannelLayoutMixin:
    """Channel layouts shared by [`WeatherNext2Config`] and [`WeatherNext2FeatureExtractor`].

    Each entry is `(variable, time offset in hours or None, number of levels)`.
    """

    input_variables: list[str] | tuple[str, ...]
    target_variables: list[str] | tuple[str, ...]
    forcing_variables: list[str] | tuple[str, ...]
    atmospheric_variables: list[str] | tuple[str, ...]
    static_variables: list[str] | tuple[str, ...]
    global_variables: list[str] | tuple[str, ...]
    pressure_levels: list[int] | tuple[int, ...]
    num_input_timesteps: int
    time_step_hours: int

    def num_levels(self, variable: str) -> int:
        return len(self.pressure_levels) if variable in self.atmospheric_variables else 1

    @property
    def past_time_offsets(self) -> list[int]:
        """Hours, relative to the initialization time, of each conditioning frame."""
        return [-(self.num_input_timesteps - 1 - i) * self.time_step_hours for i in range(self.num_input_timesteps)]

    @property
    def input_channel_layout(self) -> list[tuple[str, int | None, int]]:
        """Channels the grid encoder consumes, after the three spatial features.

        Dynamic input variables appear once per past time step, static variables once with no time,
        and forcings once at the predicted time step.
        """
        layout = []
        for variable in self.input_variables:
            if variable in self.static_variables:
                layout.append((variable, None, self.num_levels(variable)))
            else:
                layout.extend((variable, offset, self.num_levels(variable)) for offset in self.past_time_offsets)

        for variable in self.forcing_variables:
            layout.append((variable, self.time_step_hours, self.num_levels(variable)))

        return layout

    @property
    def mesh_channel_layout(self) -> list[tuple[str, int | None, int]]:
        """Channels the mesh encoder consumes, after the three spatial features.

        Only the variables with no spatial extent reach the mesh encoder directly.
        """
        return [entry for entry in self.input_channel_layout if entry[0] in self.global_variables]

    @property
    def target_channel_layout(self) -> list[tuple[str, int | None, int]]:
        return [(variable, self.time_step_hours, self.num_levels(variable)) for variable in self.target_variables]
