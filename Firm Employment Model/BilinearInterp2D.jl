# LinearInterp2D is a function that takes a produces a bilinear interpolation approximation for points off the grid of a function Z(X,Y) whose values are specified
# across a grid, X x Y.  To use LinearInterp2D the user must enter four matrices.  X, Y and Z are nx x ny matrices.  Let xvec be an nx x 1 vector of grid points along the
# dimension x.  yvec is an 1 x ny vector of grid points along the dimension y.  The matrix X is a matrix that replicates the column vector xvec ny times and concatenates
# them horizontally.  The matrix Y is a matrix that replicates the row matrix yvec nx times and concatentates them vertically.  The matrix Z contains the values of the
# function Z(X,Y) at each of the gridpoints specified by the pair (X,Y).  The user wishes to evaluate the function Z(XX,YY) at points XX and YY that may be off the
# original (X,Y) grid.  XX and YY are nxx x nyy matrices.  The matrix XX is a matrix that replicates a column vector xxvec nyy times and concatenates them horizontally.
# The matrix YY is a matrix that replicates the row matrix yyvec nxx times and concatentates them vertically.  I will follow the procedure outlined in the section
# "Interpolation in Two or More Dimensions" of the book "Numerical Recipies in C++ (2nd Ed.)"

function BilinearInterp2D(X :: Array, Y :: Array, Z :: Array, XX :: Array, YY :: Array)
	nx	= size(X,1)
	ny	= size(Y,2)
	nxx = size(XX,1)
	nyy = size(YY,2)
	xvec = X[:,1]
	# yvec = squeeze(Y[1,:],d1)    Old line v4.6
	yvec = Y[1,:]
	xxvec = XX[:,1]
	# yyvec = squeeze(YY[1,:],1)  Old line v4.6
	yyvec = YY[1,:]
	ZZ  = zeros(nxx,nyy)
	xmin  = xvec[1]
	xmax  = xvec[nx]
	ymin  = yvec[1]
	ymax  = yvec[ny]
	Zxlyl = Z[1,1]
	Zxlyh = Z[1,ny]
	Zxhyl = Z[nx,1]
	Zxhyh = Z[nx,ny]
	for j in 1 : nyy
		yj  = yyvec[j]
		lwy = searchsortedlast(dropdims(yvec',dims=1),yj)
		if lwy < ny
			yl  = yvec[lwy]
			yh  = yvec[lwy+1]
		end
		for i in 1 : nxx
			xi  = xxvec[i]
			lwx	= searchsortedlast(xvec,xi)
			if lwx < nx
				xl  = xvec[lwx]
				xh  = xvec[lwx+1]
			end
			if lwx == 1 && xi == xmin
				if lwy == 1 && yj == ymin
					ZZ[i,j] = Zxlyl
				elseif lwy == ny
					ZZ[i,j] = Zxlyh
				else
					# In this case do a piecewise linear interpolation in the direction y.
					Zh  = Z[1,lwy+1]
					Zl  = Z[1,lwy]
					slp	= (Zh - Zl)/(yh-yl);
					ZZ[i,j] = Zl + slp*(yj - yl)
				end
			elseif lwx == nx
				if lwy == 1 && yj == ymin
					ZZ[i,j] = Z[nx,1]
				elseif lwy == ny
					ZZ[i,j] = Z[nx,ny]
				else
					# In this case do a piecewise linear interpolation in the direction y.					
					Zh	= Z[nx,lwy+1]
					Zl  = Z[nx,lwy]

					slp	= (Zh - Zl)/(yh-yl);
					ZZ[i,j] = Zl + slp*(yj - yl)
				end
			else
				if lwy == 1 && yj == ymin
					Zl	= Z[lwx,1]
					Zh  = Z[lwx+1,1]
					ZZ[i,j] = Zl + (Zh - Zl)/(xh - xl)*(xi - xl)
				elseif lwy == ny
					Zl  = Z[lwx,ny]
					Zh  = Z[lwx+1,ny]
					ZZ[i,j] = Zl + (Zh - Zl)/(xh - xl)*(xi - xl)
				else
					zxlyl = Z[lwx,lwy]
					zxhyl = Z[lwx+1,lwy]
					zxlyh = Z[lwx,lwy+1]
					zxhyh = Z[lwx+1,lwy+1]

					t 	= (xi - xl)/(xh - xl)
					u   = (yj - yl)/(yh - yl)

					ZZ[i,j] = (1-t)*(1-u)*zxlyl + t*(1-u)*zxhyl + t*u*zxhyh + (1-t)*u*zxlyh
				end
			end
		end
	end
	return ZZ
end

# Employ multiple dispatch:
function BilinearInterp2D(X :: Array, Y :: Array, Z :: Array, XX :: Float64, YY :: Float64)
	nx	= size(X,1)
	ny	= size(Y,2)
	nxx = 1
	nyy = 1
	xvec = X[:,1]
	# yvec = squeeze(Y[1,:],1)    Old line v4.6
	yvec = Y[1,:]
	xxvec = XX
	# yyvec = squeeze(YY[1,:],1)  Old line v4.6
	yyvec = YY
	ZZ  = zeros(nxx,nyy)
	xmin  = xvec[1]
	xmax  = xvec[nx]
	ymin  = yvec[1]
	ymax  = yvec[ny]
	Zxlyl = Z[1,1]
	Zxlyh = Z[1,ny]
	Zxhyl = Z[nx,1]
	Zxhyh = Z[nx,ny]
	for j in 1 : nyy
		yj  = yyvec[j]
		lwy = searchsortedlast(dropdims(yvec',dims=1),yj)
		if lwy < ny
			yl  = yvec[lwy]
			yh  = yvec[lwy+1]
		end
		for i in 1 : nxx
			xi  = xxvec[i]
			lwx	= searchsortedlast(xvec,xi)
			if lwx < nx
				xl  = xvec[lwx]
				xh  = xvec[lwx+1]
			end
			if lwx == 1 && xi == xmin
				if lwy == 1 && yj == ymin
					ZZ[i,j] = Zxlyl
				elseif lwy == ny
					ZZ[i,j] = Zxlyh
				else
					# In this case do a piecewise linear interpolation in the direction y.
					Zh  = Z[1,lwy+1]
					Zl  = Z[1,lwy]
					slp	= (Zh - Zl)/(yh-yl);
					ZZ[i,j] = Zl + slp*(yj - yl)
				end
			elseif lwx == nx
				if lwy == 1 && yj == ymin
					ZZ[i,j] = Z[nx,1]
				elseif lwy == ny
					ZZ[i,j] = Z[nx,ny]
				else
					# In this case do a piecewise linear interpolation in the direction y.					
					Zh	= Z[nx,lwy+1]
					Zl  = Z[nx,lwy]

					slp	= (Zh - Zl)/(yh-yl);
					ZZ[i,j] = Zl + slp*(yj - yl)
				end
			else
				if lwy == 1 && yj == ymin
					Zl	= Z[lwx,1]
					Zh  = Z[lwx+1,1]
					ZZ[i,j] = Zl + (Zh - Zl)/(xh - xl)*(xi - xl)
				elseif lwy == ny
					Zl  = Z[lwx,ny]
					Zh  = Z[lwx+1,ny]
					ZZ[i,j] = Zl + (Zh - Zl)/(xh - xl)*(xi - xl)
				else
					zxlyl = Z[lwx,lwy]
					zxhyl = Z[lwx+1,lwy]
					zxlyh = Z[lwx,lwy+1]
					zxhyh = Z[lwx+1,lwy+1]

					t 	= (xi - xl)/(xh - xl)
					u   = (yj - yl)/(yh - yl)

					ZZ[i,j] = (1-t)*(1-u)*zxlyl + t*(1-u)*zxhyl + t*u*zxhyh + (1-t)*u*zxlyh
				end
			end
		end
	end
	ZZ = ZZ[1] 
	return ZZ
end

