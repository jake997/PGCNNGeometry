"""
Projective Group-Convolutional Neural Networks (PGCNN) Analysis Module.

This module provides tools for analyzing Projective G-CNNs using both Sage
and Macaulay2. It computes properties such as image dimension and fiber size
for group-convolutional networks with symmetric architectures.

Main class: PGCNN
    Manages network computation and algebraic geometry analysis using M2/Cremona.
"""

from sage.all import (
    var, SR, symbolic_expression, prod, vector,
    jacobian, QQ, GF, PolynomialRing
)

#from sage.rings.polynomial import PolynomialRing
from sage.interfaces.macaulay2 import macaulay2 as m2

import random
from itertools import product

from .utils import (
    compute_varphi,
    compute_Phi,
    group_matrix,
    group_action_left,
    group_action_right
)

# ===== PGCNN class ========================================================
class PGCNN:
    """Projective G-CNN helper class with Macaulay2 image/fiber analysis."""

    def __init__(self, G, r, finite_ring=False):
        """
        Initialize a Projective G-CNN network.
        
        Parameters
        ----------
        G : finite group
            The symmetry group (e.g., cyclic, dihedral, symmetric group).
        r : list of int
            Activation exponents per layer. len(r) determines the number of layers L.
        
        Attributes
        ----------
        x_vars : list
            Input signal variables [x_0, x_1, ..., x_{n-1}].
        y_vars : list
            Weight parameters organized as y_{ij} for layer i, group element j.
        params : list of list
            Weight parameters grouped by layer.
        varphi_coeffs : list
            Coefficients of the network transformation map varphi.
        Phi_coeffs : list
            Coefficients of the network output polynomial Phi.
        ring_sage : Ring
            The ring used for Sage computations (QQ or finite field).
        ring_m2 : str
            The corresponding ring specification for Macaulay2.
        """
        self.group = G
        self.r = r
        self.n = len(G.list())
        self.L = len(r)

        # symbolic variables
        self.x_vars = [var(f"x_{j}") for j in range(self.n)]
        self.y_vars = [var(f"y_{i}{j}", latex_name=f"y_{{{i}{j}}}")
                       for i in range(self.L) for j in range(self.n)]
        self.params = [self.y_vars[i*self.n:(i+1)*self.n] for i in range(self.L)]
        self.all_params = [p for layer in self.params for p in layer]

        if finite_ring:
            self.ring_sage = GF(1009)
            self.ring_m2 = "ZZ/1009"
        else: 
            self.ring_sage = QQ
            self.ring_m2 = "QQ"


        self.varphi_coeffs = None # compute varphi coefficients takes very long time, therefore it is done when needed.
        self.Phi_coeffs = compute_Phi(self.group, self.r, base_ring=self.ring_sage)




    # General helpers ------------------------------------------------------

    def _generic_point(self):
        """
        Generate a generic (random) point in parameter space with full rank.
        
        Returns
        -------
        point : list
            Flattened list of weight parameters such that all layer matrices have non-zero determinant.
        filters : list of list
            List of filters (weight vectors) for each layer.
        """
        R = self.ring_sage
        filters = [[R.random_element() for _ in self.params[i]] for i in range(self.L)]
        point = [item for filter in filters for item in filter]
        det = prod([group_matrix(list(self.group), vector(R, filter)).det() for filter in filters])

        while(det == 0):
            filters = [[R.random_element() for _ in self.params[i]] for i in range(self.L)]
            point = [item for filter in filters for item in filter]
            det = prod([group_matrix(list(self.group), vector(R, filter)).det() for filter in filters])

        return point, filters
    


    def _expected_generic_fiber(self, filters):

        R = self.ring_sage
        group_prod =[g for g in list(product(list(self.group), repeat=self.L + 1)) if g[0] == self.group.one() and g[-1] == self.group.one()]
        fiber = [[group_action_right(list(self.group), group_action_left(list(self.group), vector(R, filters[i]), g[i].inverse()), g[i+1]) 
                  for i in range(self.L) ] 
                 for g in group_prod]
        return fiber



    # Macaulay2 helpers ---------------------------------------------------
    def _init_ring_m2(self):
        """
        Initialize a multigraded polynomial ring in Macaulay2.
        
        Creates a ring where parameters are graded by layer: each parameter in
        layer i receives a degree vector with 1 in position i and 0 elsewhere.
        
        Returns
        -------
        m2 : M2 interface
            The Macaulay2 connection with ring R initialized, and the irrelevant ideal
            generated and stored as 'irr'
        """
        param_names = ", ".join(str(p) for p in self.all_params)
        
        # Create multigraded ring: each layer gets its own degree component
        degree_vectors = []
        for layer_idx in range(self.L):
            # Degree vector: 1 in position layer_idx, 0 elsewhere
            deg_vec = [0] * self.L
            deg_vec[layer_idx] = 1
            layer_size = len(self.params[layer_idx])
            # Each parameter in this layer gets the same degree vector
            for _ in range(layer_size):
                degree_vectors.append('{' + ','.join(map(str, deg_vec)) + '}')
        
        degrees_str = '{' + ', '.join(degree_vectors) + '}'
        
        # Create multigraded ring with layer-based grading
        #m2_cmd = f"R = newRing(QQ[{param_names}], Degrees => {degrees_str})"
        m2_cmd = f"R = newRing({self.ring_m2}[{param_names}], Degrees => {degrees_str})"
        m2.eval(m2_cmd)

        # Create the irrelevant ideal
        layer_ideals = []
        for i in range(self.L):
            layer_params = ",".join(str(y) for y in self.y_vars[i*self.n:(i+1)*self.n])
            layer_ideals.append(f"ideal({layer_params})")
        irr_expr = " * ".join(layer_ideals)
        result = m2.eval(f"irr = {irr_expr}")
        if "error" in result.lower():
            raise RuntimeError(f"Failed to create irr ideal: {result}")
        return m2

    def _make_map_m2(self, m2, coeffs):
        """
        Create a rational map in Macaulay2 from coefficient expressions.
        
        Converts Sage polynomial expressions to Macaulay2 format and constructs
        a rational map using the Cremona package for rational map analysis.
        
        Parameters
        ----------
        m2 : M2 interface
            Active Macaulay2 session with ring R initialized.
        coeffs : list
            Coefficient expressions (for example from compute_varphi or compute_Phi).
        
        Returns
        -------
        m2 : M2 interface
            The M2 session with the coefficients of the rational map stored as a 1-row matrix 'F',
            the kernel ideal of the rational is stored as 'baseLocus',
            and the rational map stored as 'phi'.

        """
        # switch into the ring created by _init_ring_m2
        result = m2.eval("use R")
        if "error" in result.lower():
            raise RuntimeError(f"Failed to use ring R: {result}")

        # make sure the rational‑map routines are available
        result = m2.eval('needsPackage "Cremona"')
        if "error" in result.lower():
            raise RuntimeError(f"could not load RationalMaps package: {result}")

        # turn Sage expressions into M2‑parsable strings
        coeff_strings = []
        for c in coeffs:
            coeff_str = str(c).replace("**", "^")
            coeff_strings.append(coeff_str)

        coeff_list = ", ".join(coeff_strings)

        # build a one‑row matrix; note the double braces required by M2
        m2_cmd = "F = matrix{{" + coeff_list + "}}"
        result = m2.eval(m2_cmd)
        if "error" in result.lower():
            raise RuntimeError(f"Failed to create matrix F: {result}")

        result = m2.eval("baseLocus = ideal F")
        if "error" in result.lower():
            raise RuntimeError(f"Failed to create baseLocus: {result}")

        result = m2.eval("phi = rationalMap F")
        if "error" in result.lower():
            raise RuntimeError(f"Failed to create rational map phi: {result}")

        return m2

    def _generic_image_point(self, coeffs, point=None):
        """
        Evaluate network coefficients at a generic parameter point.
        
        Parameters
        ----------
        coeffs : list
            Coefficient expressions to evaluate.
        point : list, optional
            The parameter point at which to evaluate the coefficients.
        
        Returns
        -------
        coords : str
            Comma-separated string of evaluated coordinates for M2.
        evaluated : list
            Numeric evaluated coefficients.
        """
        if point is None:
            point, _ = self._generic_point()
        sub = {self.all_params[i]: point[i] for i in range(len(self.all_params))}
        
        # Convert coefficients from polynomial ring to symbolic ring to enable substitution
        evaluated = []
        for c in coeffs:
            R = c.parent()  # polynomial ring over ring_sage

            # build evaluation hom: R → ring_sage by sending each parameter to its value in the point
            phi = R.hom(point, self.ring_sage)

            evaluated.append(phi(c))
        coords = ", ".join(str(v) for v in evaluated)
        return coords, evaluated




    def _expected_fiber_ideal_m2(self, filters=None, m2=None):
        """
        Construct the ideal of the expected fiber for given filters in Macaulay2.
        
        Builds the ideal defined by the fiber relations (F_i * p_j - F_j * p_i)
        for the expected points in the fiber based on group actions.
        
        Parameters
        ----------
        filters : list of list
            List of filters (weight vectors) for each layer.
            If None, it is initiated.
        
        m2: M2 interface
            The Macaulay2 connection with the ring 'R' and the irrelevant ideal 'irr' initiated.
            If None, then it is initiated.
        Returns
        -------
        m2 : M2 interface
            The Macaulay2 connection with the expected fiber ideal stored as 'expectedFiberIdeal'.
        """
        if m2 is None:
            m2 = self._init_ring_m2()
        
        if filters is None:
            _, filters = self._generic_point()
        # Get the expected fiber points based on group actions

        
        expected_fiber = self._expected_generic_fiber(filters)
        result = m2.eval("ideals = {}")  # Initialize an empty list to store ideals
        if "error" in result.lower():
            raise RuntimeError(f"Failed to initialize ideals list: {result}")
        for filters in expected_fiber: 
            relations = []
            for i in range(self.L):
                nonzero_index = next((i for i, val in enumerate(filters[i]) if val != 0), -1)
                for j in range(self.n):
                    if j != nonzero_index:
                        rel = f"y_{i}{j}*{filters[i][nonzero_index]}  - y_{i}{nonzero_index}*{filters[i][j]}"
                        relations.append(rel)
            if relations:
                fiber_rel_str = ", ".join(relations)
                result = m2.eval(f"fiberRel = ideal({fiber_rel_str})")
                if "error" in result.lower():
                    raise RuntimeError(f"Failed to create fiberRel ideal: {result}")
                result = m2.eval("ideals = append(ideals, fiberRel)")  # Append the ideal to the list
                if "error" in result.lower():
                    raise RuntimeError(f"Failed to append fiberRel to ideals list: {result}")
        result = m2.eval("expectedFiberIdeal = intersect ideals")  # Intersect all ideals to get the expected fiber ideal
        if "error" in result.lower():
            raise RuntimeError(f"Failed to compute expectedFiberIdeal: {result}")

        # saturate w.r.t. the irrelevant ideal
        result = m2.eval("expectedFiberIdeal = saturate(expectedFiberIdeal, irr)")
        if "error" in result.lower():
            raise RuntimeError(f"Failed to compute expectedFiberIdeal: {result}")
        
        return m2


        
    
    def _fiber_ideal_m2(self, coeffs, m2=None, point=None):
        """
        Construct the fiber ideal in Macaulay2 for a generic point.
        
        Builds the ideal defined by the fiber relations (F_i * p_j - F_j * p_i)
        with saturation with respect to the irrelevant ideal and base locus to ensure we get the correct fiber.
        
        Parameters
        ----------
        coeffs : list
            Coefficient expressions of the map.

        m2 : M2 interface
             The Macaulay2 connection with the multigraded ring 'R', the irrelevant ideal 'irr' initiated.
             If None, it is initiated.
        
        point : list
            Flattened list of weight parameters such that all layer matrices have non-zero determinant.
            Using the image of this point, we specify the fiber.

        Returns
        -------
        m2 : M2 interface
            The Macaulay2 connection with the fiber ideal stored as 'fiberIdeal'.
        """
        if m2 is None:
            m2 = self._init_ring_m2()
        m2 = self._make_map_m2(m2, coeffs)

        if point is None:
            point, _ = self._generic_point()
    
        coords, evaluated = self._generic_image_point(coeffs, point)
        nonzero_index = next((i for i, val in enumerate(evaluated) if val != 0), -1)
        result = m2.eval(f"p = {{{coords}}}")
        if "error" in result.lower():
            raise RuntimeError(f"Failed to create point p: {result}")
        
        # Create fiber relations: for each pair of coefficients, add relation F_(0,i)*p_(0,j) - F_(0,j)*p_(0,i) = 0
        # This represents the condition that the fiber contains the generic point
        relations = []
        for i in range(len(coeffs)):
            rel = f"F_(0,{nonzero_index})*p_({i}) - F_(0,{i})*p_({nonzero_index})"
            relations.append(rel)

        if relations:
            fiber_rel_str = ", ".join(relations)
            result = m2.eval(f"fiberIdeal = ideal({fiber_rel_str})")
            if "error" in result.lower():
                raise RuntimeError(f"Failed to create fiberIdeal ideal: {result}")
        else:
            # If no relations (single coefficient), create zero ideal
            result = m2.eval("fiberIdeal = ideal(0)")
            if "error" in result.lower():
                raise RuntimeError(f"Failed to create fiberIdeal ideal: {result}")
        
        result = m2.eval("fiberIdealNotSaturated = fiberIdeal")
        if "error" in result.lower():
            raise RuntimeError(f"Failed to create fiberIdeal ideal: {result}")

        result = m2.eval("fiberIdeal = saturate(fiberIdeal, baseLocus)")
        if "error" in result.lower():
                raise RuntimeError(f"Failed to saturate fiberIdeal with baseLocus: {result}")


        result = m2.eval("fiberIdeal = saturate(fiberIdeal, irr)")
        if "error" in result.lower():
                raise RuntimeError(f"Failed to saturate fiberIdeal with irr: {result}")

        return m2

        
    
    def _fiber_size_degree_m2(self, m2):
        """
        Compute generic fiber size using degree in Macaulay2
        
        Parameters
        ----------
        m2 : M2 interface           
            Macaulay2 session with fiber ideal defined as 'fiberIdeal'.
        Returns
        -------
        int
            degree of fiberIdeal
        """
        degree = m2.eval("degree fiberIdeal")
        if "error" in degree.lower():
                raise RuntimeError(f"Failed to compute degree of fiberIdeal: {degree}")
        return int(degree)

    


    # image/fiber M2 API -----------------------------------------------------
    def compute_dim_image_varphi_m2(self):
        """
        Compute the dimension of the image of the network map varphi using Macaulay2.
        
        Returns
        -------
        dim : int
            Dimension of the image of the rational map varphi.
        """
        if self.varphi_coeffs is None:
            self.varphi_coeffs = compute_varphi(self.group, self.r, base_ring=self.ring_sage)
        m2 = self._init_ring_m2()
        m2 = self._make_map_m2(m2, self.varphi_coeffs)
        result = m2.eval("dim image phi")
        if "error" in result.lower():
                raise RuntimeError(f"Failed to compute dim image phi: {result}")
        return int(result)

    def compute_dim_image_Phi_m2(self):
        """
        Compute the dimension of the image of the network polynomial Phi using Macaulay2.
        
        Returns
        -------
        dim : int
            Dimension of the image of the rational map derived from Phi.
        """
        m2 = self._init_ring_m2()
        m2 = self._make_map_m2(m2, self.Phi_coeffs)
        result = m2.eval("dim image phi")
        if "error" in result.lower():
                raise RuntimeError(f"Failed to compute dim image phi: {result}")
        return int(result)

    
    def compute_size_fiber_varphi_m2(self):
        """
        Compute the size (degree) of a generic fiber of varphi using degree of fiberIdeal.
        
        Uses multidegree saturation in Macaulay2 to compute the degree of fibers
        over generic points, accounting for the multigraded structure of the ring.
        
        Returns
        -------
        degree : int
            The generic fiber size of varphi, or -1 if the fiber is not 0-dimensional.
        """
        if self.varphi_coeffs is None:
            self.varphi_coeffs = compute_varphi(self.group, self.r, base_ring=self.ring_sage)
        point, _ = self._generic_point()
        m2 = self._fiber_ideal_m2(self.varphi_coeffs, point=point)
        degree = self._fiber_size_degree_m2(m2)
        return degree

    def compute_size_fiber_Phi_m2(self):
        """
        Compute the size (degree) of a generic fiber of Phi using using degree of fiberIdeal.
        
        Uses multidegree saturation in Macaulay2 to compute the degree of fibers
        over generic points, accounting for the multigraded structure of the ring.
        
        Returns
        -------
        degree : int
            The generic fiber size of Phi, or -1 if the fiber is not 0-dimensional.
        """
        point,_ = self._generic_point()
        m2 = self._fiber_ideal_m2(self.Phi_coeffs, point=point)
        degree = self._fiber_size_degree_m2(m2)
        return degree


    def is_generic_fiber_Phi_as_expected_m2(self):
        """
        Check if the computed fiber ideal of Phi matches the expected fiber ideal in Macaulay2.
        
        Compares the computed fiber ideal with the expected fiber ideal constructed
        from group actions. Returns True if they are equal, False otherwise.
        
        Parameters
        ----------
        m2 : M2 interface
            Macaulay2 session with both 'fiberIdeal' and 'expectedFiberIdeal' defined.
        
        Returns
        -------
        bool
            True if fiberIdeal equals expectedFiberIdeal, False otherwise.
        """

        point, filters = self._generic_point()
        m2 = self._fiber_ideal_m2(self.Phi_coeffs, point=point)
        m2 = self._expected_fiber_ideal_m2(filters, m2)
        result = m2.eval("fiberIdeal == expectedFiberIdeal")
        if "true" in result.lower():
            return True
        else:
            return False

    def is_generic_fiber_varphi_as_expected_m2(self):
        """
        Check if the computed fiber ideal of varphi matches the expected fiber ideal in Macaulay2.
        
        Compares the computed fiber ideal with the expected fiber ideal constructed
        from group actions. Returns True if they are equal, False otherwise.
        
        Parameters
        ----------
        m2 : M2 interface
            Macaulay2 session with both 'fiberIdeal' and 'expectedFiberIdeal' defined.
        
        Returns
        -------
        bool
            True if fiberIdeal equals expectedFiberIdeal, False otherwise.
        """
        if self.varphi_coeffs is None:
            self.varphi_coeffs = compute_varphi(self.group, self.r, base_ring=self.ring_sage)
        
        point, filters = self._generic_point()
        m2 = self._fiber_ideal_m2(self.varphi_coeffs, point=point)
        m2 = self._expected_fiber_ideal_m2(filters, m2)
        result = m2.eval("fiberIdeal == expectedFiberIdeal")
        if "true" in result.lower():
            return True
        else:
            return False



    # Sage helpers ---------------------------------------------------------------
    def _compute_dim_image_sage(self, coeffs):
        """
        Compute image dimension via Jacobian rank in Sage.
        
        Constructs the Jacobian matrix of the coefficient vector with respect to
        all weight parameters, evaluates at a generic point, and returns the rank.
        
        Parameters
        ----------
        coeffs : list
            Coefficient expressions to analyze.
        
        Returns
        -------
        rank : int
            Rank of the Jacobian matrix at a generic point.
        """
        gens = coeffs[0].parent().gens()  # get the generators of the polynomial ring
        J = jacobian(coeffs, gens)
        point, _ = self._generic_point()
        J = J(*point)
        rank = J.rank()
        return int(rank)

        
  
    # Sage API ---------------------------------------------------------------
    def compute_dim_image_varphi_sage(self):
        """
        Compute the dimension of the image of varphi using Sage (Jacobian method).
        
        Evaluates the Jacobian matrix at a generic point and returns its rank.
        This provides a Sage-based alternative to the Macaulay2 computation.
        
        Returns
        -------
        dim : int
            Dimension of the image (rank of Jacobian at a generic point).
        """
        if self.varphi_coeffs is None:
            self.varphi_coeffs = compute_varphi(self.group, self.r, base_ring=self.ring_sage)
        dim_image = self._compute_dim_image_sage(self.varphi_coeffs)
        return dim_image
    
    def compute_dim_image_Phi_sage(self):
        """
        Compute the dimension of the image of Phi using Sage (Jacobian method).
        
        Evaluates the Jacobian matrix at a generic point and returns its rank.
        This provides a Sage-based alternative to the Macaulay2 computation.
        
        Returns
        -------
        dim : int
            Dimension of the image (rank of Jacobian at a generic point).
        """
        dim_image = self._compute_dim_image_sage(self.Phi_coeffs)
        return dim_image




