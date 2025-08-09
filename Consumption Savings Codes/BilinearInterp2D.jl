# LinearInterp2D is a function that takes a produces a bilinear interpolation approximation for points off the grid of a function Z(X,Y) whose values are specified
# across a grid, X x Y.  To use LinearInterp2D the user must enter four matrices.  X, Y and Z are nx x ny matrices.  Let xvec be an nx x 1 vector of grid points along the
# dimension x.  yvec is an 1 x ny vector of grid points along the dimension y.  The matrix X is a matrix that replicates the column vector xvec ny times and concatenates
# them horizontally.  The matrix Y is a matrix that replicates the row matrix yvec nx times and concatentates them vertically.  The matrix Z contains the values of the
# function Z(X,Y) at each of the gridpoints specified by the pair (X,Y).  The user wishes to evaluate the function Z(XX,YY) at points XX and YY that may be off the
# original (X,Y) grid.  XX and YY are nxx x nyy matrices.  The matrix XX is a matrix that replicates a column vector xxvec nyy times and concatenates them horizontally.
# The matrix YY is a matrix that replicates the row matrix yyvec nxx times and concatentates them vertically.  I will follow the procedure outlined in the section
# "Interpolation in Two or More Dimensions" of the book "Numerical Recipies in C++ (2nd Ed.)"

function LinearInterp2D(X :: Array, Y :: Array, Z :: Array, XX :: Array, YY :: Array)
	nx	= size(X,1)
	ny	= size(Y,2)
	nxx = size(XX,1)
	nyy = size(YY,2)
	xvec = X[:,1]
	yvec = Y[1,:]
	xxvec = XX[:,1]
	yyvec = YY[1,:]
	ZZ  = zeros(nxx,nyy)
	for i in 1 : nxx
		xi  = xxvec[i]
		lwx	= searchsortedlast(xvec,xi)
		for j in 1 : nyy
			yi  = yyvec[j]
			lwy = searchsortedlast(squeeze(yvec',1),yi)
			if lwx == 1 && xi == xvec[1]
				if lwy == 1 && yi == yvec[1]
					ZZ[i,j] = Z[1,1]
				elseif lwy == ny
					ZZ[i,j] = Z[1,ny]
				else
					yl  = yvec[lwy]
					yh  = yvec[lwy+1]
					slp   = (yyvec[j] - yl)/(yh - yl)
					ZZ[i,j] = Z[1,lwy] + slp*(yi - yl)
				end
			elseif lwx == nx
				if lwy == 1 && yi == yvec[1]
					ZZ[i,j] = Z[nx,1]
				elseif lwy == ny
					ZZ[i,j] = Z[nx,ny]
				else
					yl  = yvec[lwy]
					yh  = yvec[lwy+1]

					slp   = (yyvec[j] - yl)/(yh - yl)
					ZZ[i,j] = Z[1,lwy] + slp*(yi - yl)
				end
			else
				if lwy == 1 && yi == yvec[1]
					ZZ[i,j] = Z[nx,1]
				elseif lwy == ny
					ZZ[i,j] = Z[nx,ny]
				else
					xl  = xvec[lwx]
					xh  = xvec[lwx+1]
					yl  = yvec[lwy]
					yh  = yvec[lwy+1]

					# zxlyl = Z[lwx,lwy]
					# zxhyl = Z[lwx+1,lwy]
					# zxlyh = Z[lwx,lwy+1]
					# zxhyh = Z[lwx+1,lwy+1]

					t 	= (xxvec[i] - xl)/(xh - xl)
					u   = (yyvec[j] - yl)/(yh - yl)

					ZZ[i,j] = (1-t)*(1-u)*Z[lwx,lwy] + t*(1-u)*Z[lwx+1,lwy] + t*u*Z[lwx+1,lwy+1] + (1-t)*u*Z[lwx,lwy+1]
				end
			end
		end
	end
	return ZZ
end
