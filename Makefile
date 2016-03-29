all:
	f2py2 f90perm.f90 -m f90perm -h f90perm.pyf
	f2py2 -c f90perm.pyf f90perm.f90

clean:
	rm -f ./f90perm.pyf ./f90perm.so ./*.pyc
