#rm -r build dist deepnp.egg-info
python setup.py sdist bdist_wheel \
&& twine upload dist/*
rm -r build dist deepnp.egg-info