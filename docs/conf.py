import automl_sphinx_theme
import datetime

__author__ = """LUHAI Robots"""
__email__ = """t.eimer@ai.uni-hannover.de"""
__version__ = """0.1.0"""
__copyright__ = f"Copyright {datetime.date.today().strftime('%Y')}, LUHAI Robots"


options = {
    "copyright": __copyright__,
    "author": __author__,
    "version": __version__,
    "name": "Heinrich",
    "html_theme_options": {
        "github_url": "https://github.com/theeimer/heinrich-template",
        "twitter_url": "https://twitter.com/automl_org?lang=de",
    }
}

automl_sphinx_theme.set_options(globals(), options)
