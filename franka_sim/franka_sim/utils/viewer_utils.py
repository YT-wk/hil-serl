import time

import mujoco
import mujoco.viewer


def compose_key_callbacks(*callbacks):
    callbacks = tuple(callback for callback in callbacks if callback is not None)
    if not callbacks:
        return None

    def key_callback(keycode):
        for callback in callbacks:
            callback(keycode)

    return key_callback


class DualMujocoViewer:
    def __init__(
            self,
            model,
            data,
            key_callback=None,
            num_viewers=2,
    ):
        self.model = model
        self.data = data
        self.viewer_1 = None
        self.viewer_2 = None
        self.key_callback = key_callback
        self.num_viewers = num_viewers
        self._camera_index = -1

    def __enter__(self):
        self.launch()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def launch(self):
        self.viewer_1 = mujoco.viewer.launch_passive(
            self.model,
            self.data,
            show_left_ui=False,
            show_right_ui=False,
            key_callback=self.key_callback
        )
        if self.num_viewers > 1:
            self.viewer_2 = mujoco.viewer.launch_passive(
                self.model,
                self.data,
                show_left_ui=False,
                show_right_ui=False,
                key_callback=self.key_callback
            )

    def is_running(self):
        viewers = [viewer for viewer in (self.viewer_1, self.viewer_2) if viewer]
        return bool(viewers) and all(viewer.is_running() for viewer in viewers)

    def sync(self, observation=None):
        if self.viewer_1:
            self.viewer_1.sync()
        if self.viewer_2:
            self.viewer_2.sync()

    def cycle_camera(self, direction):
        if self.model.ncam <= 0:
            print("[viewer] no fixed cameras available", flush=True)
            return

        viewers = [viewer for viewer in (self.viewer_1, self.viewer_2) if viewer]
        if not viewers:
            return

        # -1 is MuJoCo's free camera, 0..ncam-1 are fixed model cameras.
        self._camera_index += int(direction)
        if self._camera_index < -1:
            self._camera_index = self.model.ncam - 1
        elif self._camera_index >= self.model.ncam:
            self._camera_index = -1

        if self._camera_index == -1:
            camera_name = "free"
            camera_type = mujoco.mjtCamera.mjCAMERA_FREE
            fixed_camera_id = -1
        else:
            camera_name = mujoco.mj_id2name(
                self.model,
                mujoco.mjtObj.mjOBJ_CAMERA,
                self._camera_index,
            )
            camera_type = mujoco.mjtCamera.mjCAMERA_FIXED
            fixed_camera_id = self._camera_index

        for viewer in viewers:
            with viewer.lock():
                viewer.cam.type = camera_type
                viewer.cam.fixedcamid = fixed_camera_id
        print(f"[viewer] camera={camera_name}", flush=True)

    def close(self):
        viewers = [viewer for viewer in (self.viewer_1, self.viewer_2) if viewer]
        for viewer in viewers:
            viewer.close()

        deadline = time.time() + 2.0
        while time.time() < deadline:
            if all(not viewer.is_running() for viewer in viewers):
                break
            time.sleep(0.02)

        # Give MuJoCo's daemon render threads time to finish simulate.destroy()
        # before Python runs GLFW's atexit teardown.
        time.sleep(0.2)
