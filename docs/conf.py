import automl_sphinx_theme

from training_code import __copyright__, __author__, __version__


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
