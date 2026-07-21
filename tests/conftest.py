import os
import tempfile

_tmp_dir = tempfile.mkdtemp(prefix="tennis-test-")
os.environ.setdefault("ARTIFACTS_DIR", os.path.join(_tmp_dir, "artifacts"))
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_tmp_dir}/test.db")
