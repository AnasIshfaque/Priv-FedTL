import pickle
import tenseal as ts



# create a context object (private by default)

# TenSEAL context parameters
poly_modulus_degree = 8192
coeff_mod_bit_sizes = [60, 40, 40, 60]

context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree, -1, coeff_mod_bit_sizes)
context.global_scale = 2**40



# save the private context

def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)
# sample usage
save_object(context.serialize(save_secret_key=True) , 'shared_context.pkl')



# make the context public
context.make_context_public()

# save the public version of the same context
save_object(context.serialize() , 'public_context.pkl')