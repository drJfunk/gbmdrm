from distutils.core import setup


setup(
    
    name="mnSpecFit",

    
    packages = ['gbmdrm'],

    
    version = 'v0.1',
    
    description = "GBM DRM handling",
    
    author = 'J. Michael Burgess',
    
    author_email = 'jmichaelburgess@gmail.com',
    
    url = 'https://github.com/drJfunk/gbmdrm',
    keywords = ['Spectral',"GBM", "Fermi"],
    
                 
    install_requires=[
        'numpy',
        'scipy',
        'astropy',
      ])
