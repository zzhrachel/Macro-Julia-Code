function GaussHermite(n :: Int64)
    # This function determines the abscisas (x) and weights (w) for the
    # Gauss-Hermite quadrature of order n>1, on the interval [-INF, +INF].
        # This function is valid for any degree n>=2, as the companion matrix
        # (of the n'th degree Hermite polynomial) is constructed as a
        # symmetrical matrix, guaranteeing that all the eigenvalues (roots)
        # will be real.
        
    # Building the companion matrix CM
        # CM is such that det(xI-CM)=L_n(x), with L_n the Hermite polynomial
        # under consideration. Moreover, CM will be constructed in such a way
        # that it is symmetrical.
    i   = collect(1:n-1)
    a   = sqrt.(i/2)
    CM  = diagm(a,1) + diagm(a,-1)

    # Determining the abscissas (x) and weights (w)
        # - since det(xI-CM)=L_n(x), the abscissas are the roots of the
        #   characteristic polynomial, i.d. the eigenvalues of CM;
        # - the weights can be derived from the corresponding eigenvectors.
    L, V    = eig(CM)
    Li      = sortperm(L)
    #[x ind] = sort(diag(L))
    x       = L[Li]
    V       = V[:,Li]'
    w       = sqrt.(pi) * V[:,1].^2

    return x, w
end