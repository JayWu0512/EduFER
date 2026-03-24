import unittest

from edufer.core.settings import AppSettings, DEFAULT_CONFIG_PATH


class SettingsTests(unittest.TestCase):
    def test_default_config_loads(self) -> None:
        settings = AppSettings.from_file(DEFAULT_CONFIG_PATH)

        self.assertEqual(settings.app_name, "EduFER Demo")
        self.assertTrue(str(settings.detection.model_path).endswith("yolov8n-face-lindevs.onnx"))
        self.assertTrue(settings.web.static_dir.exists())


if __name__ == "__main__":
    unittest.main()
