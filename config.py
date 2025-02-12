import argparse
import sys

def parse_flags():
    # Eğer IPython ortamında çalışıyorsanız, sys.argv'yi temizleyin
    if 'ipykernel_launcher' in sys.argv[0]:
        sys.argv = ['']  # Jupyter'daki ekstra args'leri temizleriz

    # ArgumentParser başlatılır
    parser = argparse.ArgumentParser()

    # Tanımlı bayraklar (flags) #Data_224x224_garbage_6Class
    parser.add_argument('--f', type=str, default='', help='Kernel flag')
    parser.add_argument('--data_dir', type=str, default=f'Data/Data_224x224_garbage_6Class.npz', 
                        help='Directory where to find the train dataset.')
    parser.add_argument('--save_model', type=str, default='checkpoint_6Class/best_model_OurModel_HyperColumn.h5',
                        help='Save model path') #best_model_OurModel_HyperColumn
    parser.add_argument('--save_model_XGB', type=str, default='checkpoint_6Class/best_model_XGB.sav',
                        help='Save model XGB path') # best_model_XGB
    parser.add_argument('--checkpoint_fname', type=str, default='checkpoint_6Class/',
                        help='Save checkpoint path')
    parser.add_argument('--batch_size', type=int, default=128, help='Number of images to be run at the same time.')
    parser.add_argument('--epochs', type=int, default=1, help='Number of repetitions during training.')
    parser.add_argument('--number_class', type=int, default=6, help='Number of classes.')

    return parser.parse_args()



'''
# Now, use the flags
FLAGS = parse_flags()

# Example usage
print("Data Directory:", FLAGS.data_dir)
print("Model Save Path:", FLAGS.save_model)
print("Batch Size:", FLAGS.batch_size)
```
'''
'''
#import tensorflow as tf
import tensorflow.compat.v1 as tf




#### 
def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags()    
    keys_list = [keys for keys in flags_dict]    
    for keys in keys_list:
        FLAGS.__delattr__(keys)

del_all_flags(tf.flags.FLAGS)
############################!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
flags = tf.app.flags
############
tf.app.flags.DEFINE_string('f', '', 'kernel')
#####################
  
tf.app.flags.DEFINE_string('data_dir', 'Data/Data_224x224_garbage_6Class.npz',
  """ Directory where to find the train dataset.""")
  
tf.app.flags.DEFINE_string('save_model', 'checkpoint_6Class/best_model_OurModel_HyperColumn.h5',
  """save model path""")

tf.app.flags.DEFINE_string('save_model_XGB', 'checkpoint_6Class/best_model_XGB.sav',
  """save model_XGB path""")  

tf.app.flags.DEFINE_string('checkpoint_fname', 'checkpoint_6Class/',
  """save checkpoint path""") 

tf.app.flags.DEFINE_string('resuts_path', 'results/Output/',
  """save test MRI segmentation""")   


tf.app.flags.DEFINE_integer('batch_size', 128,
  """ Number of images to be run at the same time.""")
  
tf.app.flags.DEFINE_integer('epochs', 40,
  """ Number of repetitions during training.""")

tf.app.flags.DEFINE_integer('number_class', 6,
  """ Number of classes.""")

'''
