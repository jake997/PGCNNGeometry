from sage.all import (
     matrix, vector, SR, QQ, ZZ, PolynomialRing, 
    var, cartesian_product,  prod,
)





def group_matrix(G, v):
    """
    Construct a group-circulant matrix from group elements and weight vector.
    
    For a finite group G with weight vector v, creates a circulant matrix where
    the (i,j)-th entry corresponds to the weight of the group element that relates
    positions i and j. This implements group convolution in matrix form.
    
    Parameters
    ----------
    G : list
        Ordered list of elements forming a finite group.
    v : vector
        Weight vector with exactly len(G) entries, indexed by group order.
    
    Returns
    -------
    M : Matrix
        Group-circulant matrix of size len(G) × len(G) where each row/column
        corresponds to a group element.
    """
    # Precompute index mapping for all possible group products
    group_elements = list(G)
    index_map = {g: idx for idx, g in enumerate(group_elements)}
    non_zero_gs = [g for g in G if v[index_map[g]] != 0]
    n = len(group_elements)
    def entries(g):
        return [(index_map[g * h], index_map[h.inverse()]) for h in group_elements]
    
    M = matrix(v.base_ring(), n, n)
    for g in non_zero_gs:
        for i, j in entries(g):
            M[i, j] = v[index_map[g]]
    return M
   

def group_action_right(G, v, g):
    """
    Apply right group action to a vector v by group element g.
    
    For a finite group G and vector v indexed by G, computes the right action
    of g on v, resulting in a new vector where the entry corresponding to h is
    moved to the entry corresponding to h*g. This is used for convolutional
    operations in the network layers.
    
    Parameters
    ----------
    G : list
        Ordered list of elements forming a finite group.
    v : vector
        Input vector with entries indexed by group order.
    g : element of G
        Group element defining the right action.
    
    Returns
    -------
    result : vector
        New vector resulting from the right action of g on v.
    """
    index_map = {g: idx for idx, g in enumerate(G)}
    n = len(G)
    result = vector(v.base_ring(), n)
    for h in G:
        result[index_map[h]] = v[index_map[h * g]]
    return result

def group_action_left(G, v, g):
    """
    Apply left group action to a vector v by group element g.
    
    For a finite group G and vector v indexed by G, computes the left action
    of g on v, resulting in a new vector where the entry corresponding to h is
    moved to the entry corresponding to g*h. This is used for convolutional
    operations in the network layers.
    
    Parameters
    ----------
    G : list
        Ordered list of elements forming a finite group.
    v : vector
        Input vector with entries indexed by group order.
    g : element of G
        Group element defining the left action.
    
    Returns
    -------
    result : vector
        New vector resulting from the left action of g on v.
    """
    index_map = {f: idx for idx, f in enumerate(G)}
    n = len(G)
    result = vector(v.base_ring(), n)
    for h in G:
        result[index_map[h]] = v[index_map[g.inverse() * h]]
    return result

def extension(G, filter, r, R):
    """
    Extend a filter from G to G^r using group-circulant structure.
    
    Takes a filter (weight vector) over a group G and creates the associated
    group-circulant matrix for the Cartesian product G^r, which represents
    the extended convolution on multiple copies of the group.
    
    Parameters
    ----------
    G : list
        Ordered list of group elements forming a finite group.
    filter : vector or list
        Coefficients/weights for the group elements in G.
    r : int
        Number of copies (exponent): extends filter to G^r.
    R : Ring
        The base ring containing the filter elements (e.g., QQ, SR).
    
    Returns
    -------
    M : Matrix
        The group-circulant matrix of size (|G|^r) × (|G|^r) associated
        with the extended filter on G^r.
    """
    n = len(list(G))
    G_prod = cartesian_product([G for i in range(r)])

    ## Build the filter|^{G^r}.
    filter_r = [0 for i in range(n**r)]
    ns = vector(ZZ, [n**i for i in range(r)])
    for m in range(n):
        j = ns.dot_product(vector(ZZ, [m for _ in range(r)]))
        filter_r[j] = filter[m]
    
    vector_r = vector(R, filter_r)
    # print("group matrix compute ..")
    M = group_matrix(list(G_prod), vector_r)
    # print("group matrix compute done.")
    return M

def tensor_power(M, k):
    """
    Compute the k-fold tensor (Kronecker) product M ⊗ M ⊗ ... ⊗ M efficiently.
    
    Uses binary exponentiation to compute the tensor product in O(log k) matrix
    multiplications, significantly faster than naive iteration for large k.
    
    Parameters
    ----------
    M : Matrix
        A Sage matrix to be repeatedly tensor-producted.
    k : int
        Positive integer specifying the number of tensor factors.
    
    Returns
    -------
    result : Matrix
        The k-fold tensor product M ⊗ M ⊗ ... ⊗ M of size (dim(M)^k) × (dim(M)^k).
    
    Raises
    ------
    ValueError
        If k < 1.
    """
    k = int(k)
    if k < 1:
        raise ValueError("k must be >= 1")
    if k == 1:
        return M
    half_k = k//2
    rest = k - 2*half_k
    tensor_half_k = tensor_power(M, int(half_k))
    tensor_half_k = tensor_half_k.tensor_product(tensor_half_k)
    tensor_rest = tensor_power(M, rest) if rest > 0 else None
    result = tensor_half_k.tensor_product(tensor_rest) if rest > 0 else tensor_half_k
    #result = tensor_power(power2, log2).tensor_product(tensor_power(M, rest)) if rest > 0 else tensor_power(power2, log2)
    return result



def compute_varphi(G, r):
    """Compute the full network transformation as symbolic coefficients.
    
    Constructs a multi-layer group-convolutional network by composing layer-wise
    group matrices with tensor powers and extensions. Returns the coefficients
    of the network's response (first row of composed transformation matrix).
    
    Parameters
    ----------
    G : finite group
        The symmetry group (e.g., cyclic, dihedral, symmetric).
    r : list of int
        Activation parameters: repetition counts per layer. len(r) = number of layers.
    
    Returns
    -------
    coefficients : list
        Symbolic coefficients from the first row of the composed network matrix.
        These coefficients are polynomials in the weight parameters (y variables).
    """
    n = len(G.list())
    L = len(r)
    
    # Create weight parameters grouped by layer
    params = []
    for i in range(L):
        layer_params = [var(f'y_{i}{j}', latex_name=f'y_{{{i}{j}}}') for j in range(n)]
        params.append(layer_params)
    
    all_params = [p for layer in params for p in layer]
    
    # Build layer matrices using G-convolutions with tensor powers and extensions
    matrices = []
    for i in range(L):
        # print(f"Building layer {i}...")
        
        # Create G-circulant matrix for this layer's parameters
        M_circ = group_matrix(list(G), vector(SR, params[i]))
        # print(f"  Created circulant matrix")
        
        # Apply tensor power: M ⊗ M ⊗ ... ⊗ M (prod(r[i:]) times)
        M_tensor = tensor_power(M_circ, prod(r[i:]))
        # print(f"  Computed tensor power")
        
        # Extend to handle multiple group copies
        group_copies = cartesian_product([G for _ in range(prod(r[i:]))])
        M_extended = extension(group_copies, M_tensor[0], prod(r[:i]), SR)
        # print(f"  Applied extension")
        
        matrices.insert(0, M_extended)
    
    # Compose all layers: multiply matrices left to right
    varphi_matrix = matrices[0]
    for i in range(1, len(matrices)):
        varphi_matrix = varphi_matrix * matrices[i]
    
    coefficients = list(varphi_matrix[0]) # the first row of the matrix varphi include the whole filter
    return coefficients







#################################################################################################
#                                   UTILS for Phi 
##################################################################################################



def elementwise_power(V, k):
    """
    Compute the elementwise k-th power of a vector.
    
    Applies the exponent k to each entry of the vector independently.
    Used as an activation function in the network layers.
    
    Parameters
    ----------
    V : vector
        Input vector (e.g., from Sage's vector class).
    k : int
        The exponent to apply to each element.
    
    Returns
    -------
    result : vector
        Vector where each entry is the k-th power of the corresponding input entry.
    """
    return vector([V[i]**k for i in range(len(V))])

def compute_Phi(G, r):
    """Compute the network output polynomial as coefficients in weight variables.
    
    Builds a multi-layer group-convolutional network by repeatedly applying
    group-circulant matrices and elementwise power activations.
    
    Parameters
    ----------
    G : finite group
        The symmetry group defining the convolution structure.
    r : list of int
        Activation function exponents per layer. len(r) = number of layers.
    
    Returns
    -------
    coefficients : list
        Coefficients of the network polynomial Phi as polynomials in all weight
        parameters (y variables). Each coefficient is a polynomial in the weights.
    """   
    n = len(G.list())
    L = len(r)
    
    # Create weight parameters grouped by layer
    params = []
    for i in range(L):
        layer_params = [var(f'y_{i}{j}', latex_name=f'y_{{{i}{j}}}') for j in range(n)]
        params.append(layer_params)
    
    all_params = [p for layer in params for p in layer]
    
    # Create input signal variables
    x_vars = [var(f'x_{j}', latex_name=f'x_{{{j}}}') for j in range(n)]
    
    # Build network output layer by layer
    signal = vector(SR, x_vars)
    for i in range(L):
        M = group_matrix(list(G), vector(SR, params[i]))
        signal = signal * M
        signal = elementwise_power(signal, r[i])
    
    # Extract first element (the full network output)
    phi_expr = signal[0]
    
    # Convert to polynomial ring: RX[x_vars] with coefficients in RY[all_params]
    RY = PolynomialRing(QQ, all_params)
    RX = PolynomialRing(RY, x_vars)
    phi_poly = RX(phi_expr)
    
    # Extract coefficients and their corresponding monomials
    coeff_dict = phi_poly.dict()
    coefficients = list(coeff_dict.values())
    
    return coefficients
