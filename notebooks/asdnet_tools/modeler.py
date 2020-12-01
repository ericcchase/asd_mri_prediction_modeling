# requirements #
import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Conv3D, MaxPool3D, BatchNormalization, Input, Dense, Flatten,
                                     Dropout, Dropout, Add, LeakyReLU, Attention)
from tensorflow.keras.regularizers import L2
####


############################   model block classes   ###########################
class Cblock:
    def __init__(self, block_id, input_, filters, kernel_size, strides, kernel_regularizer, 
                 do_rate, padding='same'):
        x = Conv3D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, 
                   use_bias=False, kernel_regularizer=kernel_regularizer, name=f'{block_id}_conv')(input_)
        x = LeakyReLU(name=f'{block_id}_activ')(x)
        x = BatchNormalization(name=f'{block_id}_bn')(x)
        x = Dropout(rate=do_rate, name=f'{block_id}_dropout')(x)
        self.output_ = x 
        
class Dblock:
    def __init__(self, block_id, input_, units, kernel_regularizer, do_rate):    
        x = Dense(units=units, kernel_regularizer=kernel_regularizer, name=f'{block_id}_dense')(input_)
        x = LeakyReLU(name=f'{block_id}_activ')(x)
        x = BatchNormalization(name=f'{block_id}_bn')(x)
        x = Dropout(rate=do_rate, name=f'{block_id}_dropout')(x)
        self.output_ = x 
        
class Sblock:
    def __init__(self, block_id, input_wide, input_deep, filters, kernel_size, strides, kernel_regularizer, 
                 do_rate, padding='same'):
        # wide path #
        x_wide = Conv3D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, 
                   use_bias=False, kernel_regularizer=kernel_regularizer, 
                   name=f'{block_id}_conv_wide')(input_wide)
        x_wide = LeakyReLU(name=f'{block_id}_wide_activ')(x_wide)
        x_wide = BatchNormalization(name=f'{block_id}_wide_bn')(x_wide)
        x_wide = Dropout(rate=do_rate, name=f'{block_id}_wide_dropout')(x_wide)
        # add wide and deep path output #
        x = Add(name=f'{block_id}_add')([x_wide, input_deep])
        x = BatchNormalization(name=f'{block_id}_add_bn')(x)
        x = Dropout(rate=do_rate, name=f'{block_id}_add_dropout')(x)
        self.output_ = x 
        
class Ablock:
    def __init__(self, block_id, input_deep, input_wide, filters, kernel_size, strides, kernel_regularizer, 
                 do_rate, padding='valid'):
        # wide path #
        x_wide = Conv3D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, 
                   use_bias=False, kernel_regularizer=kernel_regularizer, 
                   name=f'{block_id}_conv_wide')(input_wide)
        x_wide = LeakyReLU(name=f'{block_id}_wide_activ')(x_wide)
        x_wide = BatchNormalization(name=f'{block_id}_wide_bn')(x_wide)
        x_wide = Dropout(rate=do_rate, name=f'{block_id}_wide_dropout')(x_wide)
        # add attention with the wide path output to the deep path output #
        x = Attention(name=f'{block_id}_attn',)([x_wide, input_deep])
        x = BatchNormalization(name=f'{block_id}_attn_bn')(x)
        x = Dropout(rate=do_rate, name=f'{block_id}_attn_dropout')(x)
        self.output_ = x 
        


############################   model create functions   ###########################
def create_model_skip_attention(model_id: int, params: dict, compile=True, batch_size=None, 
                                summary=False) -> keras.Model:
    '''Creates a CNN with skip blocks and an attention block.

    Args:
        model_id (int): model identifier.
        params (dict): Model Hyperparameters.
        compile (bool, optional): Defaults to True.
        batch_size (int, optional): Defaults to None.
        summary (bool, optional): Whether or not to return a model summary. Defaults to False.

    Returns:
        keras.Model
    '''
    #******************************     definition     **********************************#
    # input layer #
    input_ = Input(shape=(61, 73, 61, 1), batch_size=batch_size, name='input')
    x_stem = Conv3D(filters=16, kernel_size=1, strides=1, padding='valid', 
                    name='input_conv')(input_)

    # conv block 1 #
    cblock1 = Cblock('cblock1', x_stem, 16, 3, 1, L2(params['kreg']), params['b1do'])
    x = cblock1.output_
    # conv block 2 #
    cblock2 = Cblock('cblock2', x, 16, 3, 2, L2(params['kreg']), params['b2do'])
    x = cblock2.output_
    # conv block 3 #
    cblock3 = Cblock('cblock3', x, 16, 3, 1, L2(params['kreg']), params['b3do'])
    x = cblock3.output_
    # skip block 3 #
    sblock3 = Sblock('sblock3', x_stem, x, 16, 1, 2, L2(params['kreg']), params['b3do'])
    x = sblock3.output_    

    # conv block 4 #
    cblock4 = Cblock('cblock4', x, 32, 3, 1, L2(params['kreg']), params['b4do'])
    x = cblock4.output_
    # conv block 5 #
    cblock5 = Cblock('cblock5', x, 32, 3, 2, L2(params['kreg']), params['b5do'])
    x = cblock5.output_
    # conv block 6 #
    cblock6 = Cblock('cblock6', x, 32, 5, 1, L2(params['kreg']), params['b6do'])
    x = cblock6.output_
    # skip block 6 #
    sblock6 = Sblock('sblock6', sblock3.output_, x, 32, 1, 2, L2(params['kreg']), params['b6do'])
    x = sblock6.output_ 
    
    # conv block 7 #
    cblock7 = Cblock('cblock7', x, 64, 3, 1, L2(params['kreg']), params['b7do'])
    x = cblock7.output_
    # conv block 8 #
    cblock8 = Cblock('cblock8', x, 64, 3, 2, L2(params['kreg']), params['b8do'])
    x = cblock8.output_
   # conv block 9 #
    cblock9 = Cblock('cblock9', x, 64, 5, 1, L2(params['kreg']), params['b9do'])
    x = cblock9.output_
    # skip block 9 #
    sblock9 = Sblock('sblock9', sblock6.output_, x, 64, 1, 2, L2(params['kreg']), params['b9do'])
    x = sblock9.output_ 
    
   # conv block 10 #
    cblock10 = Cblock('cblock10', x, 128, 3, 2, L2(params['kreg']), params['b10do'])
    x = cblock10.output_
    # conv block 11 #
    cblock11 = Cblock('cblock11', x, 128, 3, 2, L2(params['kreg']), params['b11do'])
    x = cblock11.output_
   # conv block 12 #
    cblock12 = Cblock('cblock12', x, 128, 5, 2, L2(params['kreg']), params['b12do'])
    x = cblock12.output_
    # skip block 12 #
    sblock12 = Sblock('sblock12', sblock9.output_, x, 128, 1, 6, L2(params['kreg']), params['b12do'])
    x = sblock12.output_ 

    # attention block #
    ablock = Ablock('ablock', x, input_, 128, 30, 30, L2(params['kreg']), params['b12do'])
    x = ablock.output_ 


    #*******     classifier     ********
    x = Flatten()(x)

    # dense block 1 #
    dblock1 = Dblock('dblock1', x, 64, L2(params['kreg']), params['topdrop1'])
    x = dblock1.output_
    # dense block 2 #
    dblock2 = Dblock('dblock2', x, 32, L2(params['kreg']), params['topdrop2'])
    x = dblock2.output_
    
    # output #
    output_ = Dense(units=2, activation='softmax', kernel_regularizer=L2(params['kreg']))(x)



    #******************************       model       **********************************#
    m = Model(input_, output_, name=f'asdnetv6_TPU_skip_attn_{model_id}')

    if compile:
        m.compile(
        optimizer=tf.optimizers.Adam(learning_rate=.003),
        loss=tf.losses.sparse_categorical_crossentropy, 
        metrics=['accuracy']
        )
        print('model compiled!')

    print('model created!')
    if summary:
        m.summary(line_length=100) 
        print('\n# of layers:', len(m.layers))

    return m 



def create_model_shallow_sequential(model_id: int, params: dict, compile=True, batch_size=None, 
                                    summary=False) -> keras.Model:
    '''Creates a shallow, sequential CNN model.

    Args:
        model_id (int): model identifier.
        params (dict): Model Hyperparameters.
        compile (bool, optional): Defaults to True.
        batch_size ([type], optional): Defaults to None.
        summary (bool, optional): Whether or not to return a model summary. Defaults to False.

    Returns:
        keras.Model
    '''
    #******************************     definition     **********************************#
    # input layer #
    input_ = Input(shape=(61, 73, 61, 1), batch_size=batch_size, name='input')
    x_stem = Conv3D(filters=16, kernel_size=1, strides=1, padding='valid', 
               name='input_conv')(input_)

    # conv block 1 #
    cblock1 = Cblock('cblock1', x_stem, 16, 3, 1, L2(params['kreg']), params['b1do'])
    x = cblock1.output_
    # conv block 2 #
    cblock2 = Cblock('cblock2', x, 16, 3, 2, L2(params['kreg']), params['b2do'])
    x = cblock2.output_
    # conv block 3 #
    cblock3 = Cblock('cblock3', x, 16, 3, 1, L2(params['kreg']), params['b3do'])
    x = cblock3.output_
    x = MaxPool3D(2)(x) 

    # conv block 4 #
    cblock4 = Cblock('cblock4', x, 32, 3, 1, L2(params['kreg']), params['b4do'])
    x = cblock4.output_
    # conv block 5 #
    cblock5 = Cblock('cblock5', x, 32, 3, 2, L2(params['kreg']), params['b5do'])
    x = cblock5.output_
    # conv block 6 #
    cblock6 = Cblock('cblock6', x, 32, 5, 1, L2(params['kreg']), params['b6do'])
    x = cblock6.output_


    #******     classifier     *******
    x = Flatten()(x)
    
    # dense block 1 #
    dblock1 = Dblock('dblock1', x, 16, L2(params['kreg']), params['topdrop1'])
    x = dblock1.output_
    # dense block 2 #
    dblock2 = Dblock('dblock2', x, 8, L2(params['kreg']), params['topdrop2'])
    x = dblock2.output_
    
    # output #
    output_ = Dense(units=2, activation='softmax', kernel_regularizer=L2(params['kreg']))(x)




    #******************************       model       **********************************#
    m = Model(input_, output_, name=f'asdnetv6_TPU_shallow_seq_{model_id}')

    if compile:
        m.compile(
        optimizer=tf.optimizers.Adam(learning_rate=.003),
        loss=tf.losses.sparse_categorical_crossentropy, 
        metrics=['accuracy']
        )
        print('model compiled!')

    print('model created!')
    if summary:
        m.summary(line_length=100) 
        print('\n# of layers:', len(m.layers))

    return m 