# This is an example Sage script to test the PGCNN class and its methods.
from pgcnn import PGCNN

def test_PGCNN():
    G = CyclicPermutationGroup(3) #SymmetricGroup(3)  
    r = [2,  1]
    #pgcnn = PGCNN(G, r, finite_ring=True) # over a finite ring GF(1009)
    pgcnn = PGCNN(G, r) # over the rationals QQ
    print("Testing PGCNN with group:", G, "and activations:", r)    
    # Compute dimensions of images and sizes of generic fibers.
    
    dim_image_Phi_sage = pgcnn.compute_dim_image_Phi_sage()
    print(f"Dimension of image of Phi (Sage): {dim_image_Phi_sage}")

    size_fiber_Phi = pgcnn.compute_size_fiber_Phi_m2()
    print(f"Size of generic fiber of Phi (M2): {size_fiber_Phi}")

    dim_image_varphi_sage = pgcnn.compute_dim_image_varphi_sage()
    print(f"Dimension of image of varphi (Sage): {dim_image_varphi_sage}")

    size_fiber_varphi = pgcnn.compute_size_fiber_varphi_m2()
    print(f"Size of generic fiber of varphi (M2): {size_fiber_varphi}")

    is_expected_varphi = pgcnn.is_generic_fiber_varphi_as_expected_m2()
    print(f"Generic fiber of varphi is as expected (M2): {is_expected_varphi}")

    is_expected_Phi = pgcnn.is_generic_fiber_Phi_as_expected_m2()
    print(f"Generic fiber of Phi is as expected (M2): {is_expected_Phi}")

    dim_image_varphi_m2 = pgcnn.compute_dim_image_varphi_m2()
    print(f"Dimension of image of varphi (M2): {dim_image_varphi_m2}")

    dim_image_Phi_m2 = pgcnn.compute_dim_image_Phi_m2()
    print(f"Dimension of image of Phi (M2): {dim_image_Phi_m2}")



test_PGCNN()