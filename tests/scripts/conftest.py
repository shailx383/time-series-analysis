import pytest
from dotenv import load_dotenv

load_dotenv()

# Install cai-io-support and cai_secrets for local testing


@pytest.fixture
def cai_io_support():
    from cai_utils.utils import CaiIOSupport

    return CaiIOSupport()