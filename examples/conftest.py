import multiprocessing

import pytest
from playwright.sync_api import Page

# With the default method "forkserver" pytest-flask throws an error.
# "fork" is not available on all platforms (e.g. Windows).
if "fork" in multiprocessing.get_all_start_methods():
    multiprocessing.set_start_method("fork")


@pytest.fixture
def page(page: Page):
    page.set_default_navigation_timeout(3000)
    page.set_default_timeout(3000)
    return page
