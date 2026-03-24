import unittest

from edufer.app import create_app


class AppSecurityTests(unittest.TestCase):
    def test_security_middlewares_are_registered(self) -> None:
        app = create_app()
        middleware_names = [middleware.cls.__name__ for middleware in app.user_middleware]

        self.assertIn("TrustedHostMiddleware", middleware_names)
        self.assertIn("CORSMiddleware", middleware_names)
        self.assertIn("MaxRequestSizeMiddleware", middleware_names)
        self.assertIn("SecurityHeadersMiddleware", middleware_names)


if __name__ == "__main__":
    unittest.main()
