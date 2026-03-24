import os
import unittest
from unittest.mock import patch

from edufer.core.settings import AppSettings, DEFAULT_CONFIG_PATH


class SettingsTests(unittest.TestCase):
    def test_default_config_loads(self) -> None:
        settings = AppSettings.from_file(DEFAULT_CONFIG_PATH)

        self.assertEqual(settings.app_name, "EduFER Demo")
        self.assertTrue(str(settings.detection.model_path).endswith("yolov8n-face-lindevs.onnx"))
        self.assertTrue(settings.web.static_dir.exists())
        self.assertIn("http://localhost:8000", settings.security.cors_allowed_origins)
        self.assertIn("localhost", settings.security.trusted_hosts)

    def test_env_overrides_apply(self) -> None:
        with patch.dict(
            os.environ,
            {
                "EDUFER_PORT": "9000",
                "EDUFER_ALLOWED_ORIGINS": "https://demo.example.com",
                "EDUFER_TRUSTED_HOSTS": "demo.example.com",
                "EDUFER_ENABLE_HTTPS_REDIRECT": "true",
            },
            clear=False,
        ):
            settings = AppSettings.from_file(DEFAULT_CONFIG_PATH)

        self.assertEqual(settings.port, 9000)
        self.assertEqual(settings.security.cors_allowed_origins, ["https://demo.example.com"])
        self.assertEqual(settings.security.trusted_hosts, ["demo.example.com"])
        self.assertTrue(settings.security.enable_https_redirect)


if __name__ == "__main__":
    unittest.main()
