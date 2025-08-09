function monthlytoquarterly(x :: Array{Float64,1})

    # PURPOSE : This function takes data that is in monthly form and takes the 3 month average.
    #-------------------------------------------------------------------------------------------------------------------------------------
    # USAGE : The user enters the following
    #
    #       x = the matrix of monthly data
    #
    #-------------------------------------------------------------------------------------------------------------------------------------

    obs     = size(x,1)
    obsrem  = rem(obs,3)
    if obsrem !== 0
        return error("There are an inappropriate number of months in the dataset")
    end

    quarterly   = []
    counter = 1
    while counter < obs
        xt1     = x[counter]
        xt2     = x[counter+1]
        xt3     = x[counter+2]
        xsum    = xt1 + xt2 + xt3
        xave    = xsum/3
        quarterly = vcat(quarterly,xave)
        counter += 3
    end
    return quarterly
end
   

function monthlytoquarterly(x :: Vector)

    # PURPOSE : This function takes data that is in monthly form and takes the 3 month average.
    #-------------------------------------------------------------------------------------------------------------------------------------
    # USAGE : The user enters the following
    #
    #       x = the matrix of monthly data
    #
    #-------------------------------------------------------------------------------------------------------------------------------------

    obs     = size(x,1)
    obsrem  = rem(obs,3)
    if obsrem !== 0
        return error("There are an inappropriate number of months in the dataset")
    end

    quarterly   = []
    counter = 1
    while counter < obs
        xt1     = x[counter]
        xt2     = x[counter+1]
        xt3     = x[counter+2]
        xsum    = xt1 + xt2 + xt3
        xave    = xsum/3
        quarterly = vcat(quarterly,xave)
        counter += 3
    end
    return quarterly
end    
    