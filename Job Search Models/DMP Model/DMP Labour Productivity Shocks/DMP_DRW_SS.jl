function DMP_DRW_SS(x0 :: Real, Param :: Parameter_Values)

	# This function solves for the steady state v-u ratio.
	rho     = Param.rho
	beta    = Param.beta
	delta   = Param.delta
	alpha   = Param.alpha
	z       = Param.z
	mA      = Param.mA
	b       = Param.b

	thss    = x0
	qss     = 1/((1+thss^alpha)^(1/alpha))
	wss     = beta*exp(mA) + (1-beta)*b + beta*z*thss
	Jss     = (exp(mA)-wss)/(1-rho*(1-delta))

	res     = z - rho*qss*Jss
	return res
end