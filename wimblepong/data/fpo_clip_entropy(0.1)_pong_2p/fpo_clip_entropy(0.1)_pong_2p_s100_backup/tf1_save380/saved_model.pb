��(
�'�&
:
Add
x"T
y"T
z"T"
Ttype:
2	
W
AddN
inputs"T*N
sum"T"
Nint(0"!
Ttype:
2	��
�
	ApplyAdam
var"T�	
m"T�	
v"T�
beta1_power"T
beta2_power"T
lr"T

beta1"T

beta2"T
epsilon"T	
grad"T
out"T�" 
Ttype:
2	"
use_lockingbool( "
use_nesterovbool( 
x
Assign
ref"T�

value"T

output_ref"T�"	
Ttype"
validate_shapebool("
use_lockingbool(�
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
~
BiasAddGrad
out_backprop"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
S
DynamicStitch
indices*N
data"T*N
merged"T"
Nint(0"	
Ttype
,
Exp
x"T
y"T"
Ttype:

2
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
?
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
9
FloorMod
x"T
y"T
z"T"
Ttype:

2	
=
Greater
x"T
y"T
z
"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
:
Less
x"T
y"T
z
"
Ttype:
2	
?
	LessEqual
x"T
y"T
z
"
Ttype:
2	
,
Log
x"T
y"T"
Ttype:

2
?

LogSoftmax
logits"T

logsoftmax"T"
Ttype:
2
#
	LogicalOr
x

y

z
�
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
8
Maximum
x"T
y"T
z"T"
Ttype:

2	
�
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�
8
Minimum
x"T
y"T
z"T"
Ttype:

2	
=
Mul
x"T
y"T
z"T"
Ttype:
2	�
�
Multinomial
logits"T
num_samples
output"output_dtype"
seedint "
seed2int "
Ttype:
2	" 
output_dtypetype0	:
2	�
.
Neg
x"T
y"T"
Ttype:

2	

NoOp
�
OneHot
indices"TI	
depth
on_value"T
	off_value"T
output"T"
axisint���������"	
Ttype"
TItype0	:
2	
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape
6
Pow
x"T
y"T
z"T"
Ttype:

2	
�
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
PyFunc
input2Tin
output2Tout"
tokenstring"
Tin
list(type)("
Tout
list(type)(�
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	�
a
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:	
2	
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
�
SplitV

value"T
size_splits"Tlen
	split_dim
output"T*	num_split"
	num_splitint(0"	
Ttype"
Tlentype0	:
2	
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
:
Sub
x"T
y"T
z"T"
Ttype:
2	
�
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
-
Tanh
x"T
y"T"
Ttype:

2
:
TanhGrad
y"T
dy"T
z"T"
Ttype:

2
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
s

VariableV2
ref"dtype�"
shapeshape"
dtypetype"
	containerstring "
shared_namestring �
&
	ZerosLike
x"T
y"T"	
Ttype"serve*1.14.02unknown��'
n
PlaceholderPlaceholder*
dtype0*'
_output_shapes
:���������*
shape:���������
h
Placeholder_1Placeholder*
dtype0*#
_output_shapes
:���������*
shape:���������
h
Placeholder_2Placeholder*#
_output_shapes
:���������*
shape:���������*
dtype0
h
Placeholder_3Placeholder*
shape:���������*
dtype0*#
_output_shapes
:���������
h
Placeholder_4Placeholder*
shape:���������*
dtype0*#
_output_shapes
:���������
�
0pi/dense/kernel/Initializer/random_uniform/shapeConst*"
_class
loc:@pi/dense/kernel*
dtype0*
_output_shapes
:*
valueB"   @   
�
.pi/dense/kernel/Initializer/random_uniform/minConst*"
_class
loc:@pi/dense/kernel*
valueB
 *�啾*
dtype0*
_output_shapes
: 
�
.pi/dense/kernel/Initializer/random_uniform/maxConst*
valueB
 *��>*
_output_shapes
: *"
_class
loc:@pi/dense/kernel*
dtype0
�
8pi/dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform0pi/dense/kernel/Initializer/random_uniform/shape*
T0*
dtype0*

seedd*
_output_shapes

:@*
seed2*"
_class
loc:@pi/dense/kernel
�
.pi/dense/kernel/Initializer/random_uniform/subSub.pi/dense/kernel/Initializer/random_uniform/max.pi/dense/kernel/Initializer/random_uniform/min*
T0*
_output_shapes
: *"
_class
loc:@pi/dense/kernel
�
.pi/dense/kernel/Initializer/random_uniform/mulMul8pi/dense/kernel/Initializer/random_uniform/RandomUniform.pi/dense/kernel/Initializer/random_uniform/sub*
_output_shapes

:@*"
_class
loc:@pi/dense/kernel*
T0
�
*pi/dense/kernel/Initializer/random_uniformAdd.pi/dense/kernel/Initializer/random_uniform/mul.pi/dense/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@pi/dense/kernel*
_output_shapes

:@
�
pi/dense/kernel
VariableV2*
shape
:@*
	container *
shared_name *
_output_shapes

:@*"
_class
loc:@pi/dense/kernel*
dtype0
�
pi/dense/kernel/AssignAssignpi/dense/kernel*pi/dense/kernel/Initializer/random_uniform*
validate_shape(*
T0*
_output_shapes

:@*
use_locking(*"
_class
loc:@pi/dense/kernel
~
pi/dense/kernel/readIdentitypi/dense/kernel*
T0*
_output_shapes

:@*"
_class
loc:@pi/dense/kernel
�
pi/dense/bias/Initializer/zerosConst*
dtype0*
_output_shapes
:@* 
_class
loc:@pi/dense/bias*
valueB@*    
�
pi/dense/bias
VariableV2*
_output_shapes
:@*
shape:@*
shared_name * 
_class
loc:@pi/dense/bias*
	container *
dtype0
�
pi/dense/bias/AssignAssignpi/dense/biaspi/dense/bias/Initializer/zeros*
validate_shape(*
_output_shapes
:@*
T0* 
_class
loc:@pi/dense/bias*
use_locking(
t
pi/dense/bias/readIdentitypi/dense/bias*
_output_shapes
:@*
T0* 
_class
loc:@pi/dense/bias
�
pi/dense/MatMulMatMulPlaceholderpi/dense/kernel/read*
transpose_a( *
T0*
transpose_b( *'
_output_shapes
:���������@
�
pi/dense/BiasAddBiasAddpi/dense/MatMulpi/dense/bias/read*
T0*'
_output_shapes
:���������@*
data_formatNHWC
Y
pi/dense/TanhTanhpi/dense/BiasAdd*
T0*'
_output_shapes
:���������@
�
2pi/dense_1/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*
valueB"@   @   *
dtype0*$
_class
loc:@pi/dense_1/kernel
�
0pi/dense_1/kernel/Initializer/random_uniform/minConst*
_output_shapes
: *
valueB
 *׳]�*$
_class
loc:@pi/dense_1/kernel*
dtype0
�
0pi/dense_1/kernel/Initializer/random_uniform/maxConst*$
_class
loc:@pi/dense_1/kernel*
dtype0*
_output_shapes
: *
valueB
 *׳]>
�
:pi/dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform2pi/dense_1/kernel/Initializer/random_uniform/shape*
dtype0*

seedd*
T0*$
_class
loc:@pi/dense_1/kernel*
_output_shapes

:@@*
seed2
�
0pi/dense_1/kernel/Initializer/random_uniform/subSub0pi/dense_1/kernel/Initializer/random_uniform/max0pi/dense_1/kernel/Initializer/random_uniform/min*$
_class
loc:@pi/dense_1/kernel*
T0*
_output_shapes
: 
�
0pi/dense_1/kernel/Initializer/random_uniform/mulMul:pi/dense_1/kernel/Initializer/random_uniform/RandomUniform0pi/dense_1/kernel/Initializer/random_uniform/sub*$
_class
loc:@pi/dense_1/kernel*
T0*
_output_shapes

:@@
�
,pi/dense_1/kernel/Initializer/random_uniformAdd0pi/dense_1/kernel/Initializer/random_uniform/mul0pi/dense_1/kernel/Initializer/random_uniform/min*
_output_shapes

:@@*
T0*$
_class
loc:@pi/dense_1/kernel
�
pi/dense_1/kernel
VariableV2*
	container *
shared_name *$
_class
loc:@pi/dense_1/kernel*
dtype0*
_output_shapes

:@@*
shape
:@@
�
pi/dense_1/kernel/AssignAssignpi/dense_1/kernel,pi/dense_1/kernel/Initializer/random_uniform*
use_locking(*$
_class
loc:@pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@*
T0
�
pi/dense_1/kernel/readIdentitypi/dense_1/kernel*
_output_shapes

:@@*$
_class
loc:@pi/dense_1/kernel*
T0
�
!pi/dense_1/bias/Initializer/zerosConst*"
_class
loc:@pi/dense_1/bias*
dtype0*
_output_shapes
:@*
valueB@*    
�
pi/dense_1/bias
VariableV2*"
_class
loc:@pi/dense_1/bias*
shared_name *
shape:@*
_output_shapes
:@*
	container *
dtype0
�
pi/dense_1/bias/AssignAssignpi/dense_1/bias!pi/dense_1/bias/Initializer/zeros*"
_class
loc:@pi/dense_1/bias*
T0*
_output_shapes
:@*
use_locking(*
validate_shape(
z
pi/dense_1/bias/readIdentitypi/dense_1/bias*"
_class
loc:@pi/dense_1/bias*
T0*
_output_shapes
:@
�
pi/dense_1/MatMulMatMulpi/dense/Tanhpi/dense_1/kernel/read*
transpose_a( *'
_output_shapes
:���������@*
transpose_b( *
T0
�
pi/dense_1/BiasAddBiasAddpi/dense_1/MatMulpi/dense_1/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:���������@
]
pi/dense_1/TanhTanhpi/dense_1/BiasAdd*'
_output_shapes
:���������@*
T0
�
2pi/dense_2/kernel/Initializer/random_uniform/shapeConst*
valueB"@      *
_output_shapes
:*
dtype0*$
_class
loc:@pi/dense_2/kernel
�
0pi/dense_2/kernel/Initializer/random_uniform/minConst*
_output_shapes
: *$
_class
loc:@pi/dense_2/kernel*
dtype0*
valueB
 *�7��
�
0pi/dense_2/kernel/Initializer/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *�7�>*$
_class
loc:@pi/dense_2/kernel
�
:pi/dense_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform2pi/dense_2/kernel/Initializer/random_uniform/shape*
T0*
dtype0*
_output_shapes

:@*
seed2**

seedd*$
_class
loc:@pi/dense_2/kernel
�
0pi/dense_2/kernel/Initializer/random_uniform/subSub0pi/dense_2/kernel/Initializer/random_uniform/max0pi/dense_2/kernel/Initializer/random_uniform/min*
_output_shapes
: *$
_class
loc:@pi/dense_2/kernel*
T0
�
0pi/dense_2/kernel/Initializer/random_uniform/mulMul:pi/dense_2/kernel/Initializer/random_uniform/RandomUniform0pi/dense_2/kernel/Initializer/random_uniform/sub*$
_class
loc:@pi/dense_2/kernel*
T0*
_output_shapes

:@
�
,pi/dense_2/kernel/Initializer/random_uniformAdd0pi/dense_2/kernel/Initializer/random_uniform/mul0pi/dense_2/kernel/Initializer/random_uniform/min*$
_class
loc:@pi/dense_2/kernel*
T0*
_output_shapes

:@
�
pi/dense_2/kernel
VariableV2*
_output_shapes

:@*
dtype0*
	container *
shared_name *
shape
:@*$
_class
loc:@pi/dense_2/kernel
�
pi/dense_2/kernel/AssignAssignpi/dense_2/kernel,pi/dense_2/kernel/Initializer/random_uniform*
validate_shape(*
T0*
_output_shapes

:@*
use_locking(*$
_class
loc:@pi/dense_2/kernel
�
pi/dense_2/kernel/readIdentitypi/dense_2/kernel*
_output_shapes

:@*
T0*$
_class
loc:@pi/dense_2/kernel
�
!pi/dense_2/bias/Initializer/zerosConst*"
_class
loc:@pi/dense_2/bias*
dtype0*
valueB*    *
_output_shapes
:
�
pi/dense_2/bias
VariableV2*
shared_name *
	container *
_output_shapes
:*"
_class
loc:@pi/dense_2/bias*
shape:*
dtype0
�
pi/dense_2/bias/AssignAssignpi/dense_2/bias!pi/dense_2/bias/Initializer/zeros*"
_class
loc:@pi/dense_2/bias*
use_locking(*
T0*
_output_shapes
:*
validate_shape(
z
pi/dense_2/bias/readIdentitypi/dense_2/bias*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:*
T0
�
pi/dense_2/MatMulMatMulpi/dense_1/Tanhpi/dense_2/kernel/read*
transpose_a( *
T0*'
_output_shapes
:���������*
transpose_b( 
�
pi/dense_2/BiasAddBiasAddpi/dense_2/MatMulpi/dense_2/bias/read*'
_output_shapes
:���������*
T0*
data_formatNHWC
a
pi/LogSoftmax
LogSoftmaxpi/dense_2/BiasAdd*
T0*'
_output_shapes
:���������
h
&pi/multinomial/Multinomial/num_samplesConst*
dtype0*
_output_shapes
: *
value	B :
�
pi/multinomial/MultinomialMultinomialpi/dense_2/BiasAdd&pi/multinomial/Multinomial/num_samples*
seed28*'
_output_shapes
:���������*

seedd*
T0*
output_dtype0	
v

pi/SqueezeSqueezepi/multinomial/Multinomial*
squeeze_dims
*
T0	*#
_output_shapes
:���������
X
pi/one_hot/on_valueConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
Y
pi/one_hot/off_valueConst*
valueB
 *    *
_output_shapes
: *
dtype0
R
pi/one_hot/depthConst*
dtype0*
_output_shapes
: *
value	B :
�

pi/one_hotOneHotPlaceholder_1pi/one_hot/depthpi/one_hot/on_valuepi/one_hot/off_value*
T0*
axis���������*
TI0*'
_output_shapes
:���������
Z
pi/mulMul
pi/one_hotpi/LogSoftmax*
T0*'
_output_shapes
:���������
Z
pi/Sum/reduction_indicesConst*
value	B :*
_output_shapes
: *
dtype0
z
pi/SumSumpi/mulpi/Sum/reduction_indices*
T0*

Tidx0*
	keep_dims( *#
_output_shapes
:���������
Z
pi/one_hot_1/on_valueConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
[
pi/one_hot_1/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    
T
pi/one_hot_1/depthConst*
_output_shapes
: *
dtype0*
value	B :
�
pi/one_hot_1OneHot
pi/Squeezepi/one_hot_1/depthpi/one_hot_1/on_valuepi/one_hot_1/off_value*'
_output_shapes
:���������*
T0*
TI0	*
axis���������
^
pi/mul_1Mulpi/one_hot_1pi/LogSoftmax*
T0*'
_output_shapes
:���������
\
pi/Sum_1/reduction_indicesConst*
value	B :*
_output_shapes
: *
dtype0
�
pi/Sum_1Sumpi/mul_1pi/Sum_1/reduction_indices*
T0*

Tidx0*#
_output_shapes
:���������*
	keep_dims( 
�
/v/dense/kernel/Initializer/random_uniform/shapeConst*!
_class
loc:@v/dense/kernel*
_output_shapes
:*
valueB"   @   *
dtype0
�
-v/dense/kernel/Initializer/random_uniform/minConst*
_output_shapes
: *
valueB
 *�啾*
dtype0*!
_class
loc:@v/dense/kernel
�
-v/dense/kernel/Initializer/random_uniform/maxConst*
_output_shapes
: *
dtype0*!
_class
loc:@v/dense/kernel*
valueB
 *��>
�
7v/dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform/v/dense/kernel/Initializer/random_uniform/shape*
_output_shapes

:@*
T0*

seedd*
seed2L*!
_class
loc:@v/dense/kernel*
dtype0
�
-v/dense/kernel/Initializer/random_uniform/subSub-v/dense/kernel/Initializer/random_uniform/max-v/dense/kernel/Initializer/random_uniform/min*
T0*!
_class
loc:@v/dense/kernel*
_output_shapes
: 
�
-v/dense/kernel/Initializer/random_uniform/mulMul7v/dense/kernel/Initializer/random_uniform/RandomUniform-v/dense/kernel/Initializer/random_uniform/sub*
_output_shapes

:@*
T0*!
_class
loc:@v/dense/kernel
�
)v/dense/kernel/Initializer/random_uniformAdd-v/dense/kernel/Initializer/random_uniform/mul-v/dense/kernel/Initializer/random_uniform/min*
_output_shapes

:@*
T0*!
_class
loc:@v/dense/kernel
�
v/dense/kernel
VariableV2*
_output_shapes

:@*
dtype0*
shape
:@*!
_class
loc:@v/dense/kernel*
	container *
shared_name 
�
v/dense/kernel/AssignAssignv/dense/kernel)v/dense/kernel/Initializer/random_uniform*
use_locking(*
_output_shapes

:@*
validate_shape(*
T0*!
_class
loc:@v/dense/kernel
{
v/dense/kernel/readIdentityv/dense/kernel*!
_class
loc:@v/dense/kernel*
_output_shapes

:@*
T0
�
v/dense/bias/Initializer/zerosConst*
valueB@*    *
_class
loc:@v/dense/bias*
dtype0*
_output_shapes
:@
�
v/dense/bias
VariableV2*
shared_name *
_class
loc:@v/dense/bias*
shape:@*
_output_shapes
:@*
dtype0*
	container 
�
v/dense/bias/AssignAssignv/dense/biasv/dense/bias/Initializer/zeros*
use_locking(*
_output_shapes
:@*
_class
loc:@v/dense/bias*
T0*
validate_shape(
q
v/dense/bias/readIdentityv/dense/bias*
T0*
_output_shapes
:@*
_class
loc:@v/dense/bias
�
v/dense/MatMulMatMulPlaceholderv/dense/kernel/read*
transpose_a( *
transpose_b( *'
_output_shapes
:���������@*
T0
�
v/dense/BiasAddBiasAddv/dense/MatMulv/dense/bias/read*
T0*'
_output_shapes
:���������@*
data_formatNHWC
W
v/dense/TanhTanhv/dense/BiasAdd*
T0*'
_output_shapes
:���������@
�
1v/dense_1/kernel/Initializer/random_uniform/shapeConst*
valueB"@   @   *#
_class
loc:@v/dense_1/kernel*
dtype0*
_output_shapes
:
�
/v/dense_1/kernel/Initializer/random_uniform/minConst*
dtype0*#
_class
loc:@v/dense_1/kernel*
_output_shapes
: *
valueB
 *׳]�
�
/v/dense_1/kernel/Initializer/random_uniform/maxConst*
valueB
 *׳]>*
_output_shapes
: *
dtype0*#
_class
loc:@v/dense_1/kernel
�
9v/dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform1v/dense_1/kernel/Initializer/random_uniform/shape*
seed2]*
_output_shapes

:@@*
dtype0*

seedd*#
_class
loc:@v/dense_1/kernel*
T0
�
/v/dense_1/kernel/Initializer/random_uniform/subSub/v/dense_1/kernel/Initializer/random_uniform/max/v/dense_1/kernel/Initializer/random_uniform/min*
T0*
_output_shapes
: *#
_class
loc:@v/dense_1/kernel
�
/v/dense_1/kernel/Initializer/random_uniform/mulMul9v/dense_1/kernel/Initializer/random_uniform/RandomUniform/v/dense_1/kernel/Initializer/random_uniform/sub*
_output_shapes

:@@*#
_class
loc:@v/dense_1/kernel*
T0
�
+v/dense_1/kernel/Initializer/random_uniformAdd/v/dense_1/kernel/Initializer/random_uniform/mul/v/dense_1/kernel/Initializer/random_uniform/min*#
_class
loc:@v/dense_1/kernel*
T0*
_output_shapes

:@@
�
v/dense_1/kernel
VariableV2*
_output_shapes

:@@*
shape
:@@*#
_class
loc:@v/dense_1/kernel*
shared_name *
dtype0*
	container 
�
v/dense_1/kernel/AssignAssignv/dense_1/kernel+v/dense_1/kernel/Initializer/random_uniform*
T0*
_output_shapes

:@@*#
_class
loc:@v/dense_1/kernel*
validate_shape(*
use_locking(
�
v/dense_1/kernel/readIdentityv/dense_1/kernel*
T0*#
_class
loc:@v/dense_1/kernel*
_output_shapes

:@@
�
 v/dense_1/bias/Initializer/zerosConst*
_output_shapes
:@*!
_class
loc:@v/dense_1/bias*
dtype0*
valueB@*    
�
v/dense_1/bias
VariableV2*
dtype0*!
_class
loc:@v/dense_1/bias*
	container *
_output_shapes
:@*
shared_name *
shape:@
�
v/dense_1/bias/AssignAssignv/dense_1/bias v/dense_1/bias/Initializer/zeros*
use_locking(*
validate_shape(*
T0*!
_class
loc:@v/dense_1/bias*
_output_shapes
:@
w
v/dense_1/bias/readIdentityv/dense_1/bias*!
_class
loc:@v/dense_1/bias*
T0*
_output_shapes
:@
�
v/dense_1/MatMulMatMulv/dense/Tanhv/dense_1/kernel/read*
transpose_b( *
T0*'
_output_shapes
:���������@*
transpose_a( 
�
v/dense_1/BiasAddBiasAddv/dense_1/MatMulv/dense_1/bias/read*
T0*'
_output_shapes
:���������@*
data_formatNHWC
[
v/dense_1/TanhTanhv/dense_1/BiasAdd*'
_output_shapes
:���������@*
T0
�
1v/dense_2/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*
dtype0*
valueB"@      *#
_class
loc:@v/dense_2/kernel
�
/v/dense_2/kernel/Initializer/random_uniform/minConst*
dtype0*
valueB
 *����*#
_class
loc:@v/dense_2/kernel*
_output_shapes
: 
�
/v/dense_2/kernel/Initializer/random_uniform/maxConst*#
_class
loc:@v/dense_2/kernel*
dtype0*
_output_shapes
: *
valueB
 *���>
�
9v/dense_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform1v/dense_2/kernel/Initializer/random_uniform/shape*
seed2n*
_output_shapes

:@*

seedd*#
_class
loc:@v/dense_2/kernel*
T0*
dtype0
�
/v/dense_2/kernel/Initializer/random_uniform/subSub/v/dense_2/kernel/Initializer/random_uniform/max/v/dense_2/kernel/Initializer/random_uniform/min*#
_class
loc:@v/dense_2/kernel*
T0*
_output_shapes
: 
�
/v/dense_2/kernel/Initializer/random_uniform/mulMul9v/dense_2/kernel/Initializer/random_uniform/RandomUniform/v/dense_2/kernel/Initializer/random_uniform/sub*
_output_shapes

:@*#
_class
loc:@v/dense_2/kernel*
T0
�
+v/dense_2/kernel/Initializer/random_uniformAdd/v/dense_2/kernel/Initializer/random_uniform/mul/v/dense_2/kernel/Initializer/random_uniform/min*
T0*
_output_shapes

:@*#
_class
loc:@v/dense_2/kernel
�
v/dense_2/kernel
VariableV2*
shared_name *
dtype0*
	container *
shape
:@*
_output_shapes

:@*#
_class
loc:@v/dense_2/kernel
�
v/dense_2/kernel/AssignAssignv/dense_2/kernel+v/dense_2/kernel/Initializer/random_uniform*
_output_shapes

:@*
validate_shape(*
use_locking(*#
_class
loc:@v/dense_2/kernel*
T0
�
v/dense_2/kernel/readIdentityv/dense_2/kernel*
_output_shapes

:@*#
_class
loc:@v/dense_2/kernel*
T0
�
 v/dense_2/bias/Initializer/zerosConst*!
_class
loc:@v/dense_2/bias*
dtype0*
_output_shapes
:*
valueB*    
�
v/dense_2/bias
VariableV2*
shape:*
dtype0*
_output_shapes
:*
shared_name *
	container *!
_class
loc:@v/dense_2/bias
�
v/dense_2/bias/AssignAssignv/dense_2/bias v/dense_2/bias/Initializer/zeros*
_output_shapes
:*
T0*
validate_shape(*
use_locking(*!
_class
loc:@v/dense_2/bias
w
v/dense_2/bias/readIdentityv/dense_2/bias*!
_class
loc:@v/dense_2/bias*
T0*
_output_shapes
:
�
v/dense_2/MatMulMatMulv/dense_1/Tanhv/dense_2/kernel/read*'
_output_shapes
:���������*
T0*
transpose_a( *
transpose_b( 
�
v/dense_2/BiasAddBiasAddv/dense_2/MatMulv/dense_2/bias/read*
data_formatNHWC*
T0*'
_output_shapes
:���������
l
	v/SqueezeSqueezev/dense_2/BiasAdd*
squeeze_dims
*#
_output_shapes
:���������*
T0
O
subSubpi/SumPlaceholder_4*
T0*#
_output_shapes
:���������
=
ExpExpsub*
T0*#
_output_shapes
:���������
N
	Greater/yConst*
dtype0*
valueB
 *    *
_output_shapes
: 
Z
GreaterGreaterPlaceholder_2	Greater/y*
T0*#
_output_shapes
:���������
J
mul/xConst*
dtype0*
_output_shapes
: *
valueB
 *���?
N
mulMulmul/xPlaceholder_2*#
_output_shapes
:���������*
T0
L
mul_1/xConst*
valueB
 *��L?*
_output_shapes
: *
dtype0
R
mul_1Mulmul_1/xPlaceholder_2*
T0*#
_output_shapes
:���������
S
SelectSelectGreatermulmul_1*
T0*#
_output_shapes
:���������
N
mul_2MulExpPlaceholder_2*#
_output_shapes
:���������*
T0
O
MinimumMinimummul_2Select*
T0*#
_output_shapes
:���������
O
ConstConst*
_output_shapes
:*
valueB: *
dtype0
Z
MeanMeanMinimumConst*

Tidx0*
T0*
_output_shapes
: *
	keep_dims( 
1
NegNegMean*
T0*
_output_shapes
: 
T
sub_1SubPlaceholder_3	v/Squeeze*
T0*#
_output_shapes
:���������
J
pow/yConst*
valueB
 *   @*
dtype0*
_output_shapes
: 
F
powPowsub_1pow/y*
T0*#
_output_shapes
:���������
Q
Const_1Const*
valueB: *
_output_shapes
:*
dtype0
Z
Mean_1MeanpowConst_1*

Tidx0*
_output_shapes
: *
T0*
	keep_dims( 
Q
sub_2SubPlaceholder_4pi/Sum*#
_output_shapes
:���������*
T0
Q
Const_2Const*
valueB: *
_output_shapes
:*
dtype0
\
Mean_2Meansub_2Const_2*

Tidx0*
_output_shapes
: *
	keep_dims( *
T0
M
Neg_1Negpi/LogSoftmax*'
_output_shapes
:���������*
T0
M
Exp_1Exppi/LogSoftmax*'
_output_shapes
:���������*
T0
L
mul_3MulNeg_1Exp_1*
T0*'
_output_shapes
:���������
X
Const_3Const*
dtype0*
_output_shapes
:*
valueB"       
\
Mean_3Meanmul_3Const_3*
	keep_dims( *
T0*

Tidx0*
_output_shapes
: 
L
mul_4/yConst*
valueB
 *  @@*
dtype0*
_output_shapes
: 
>
mul_4MulMean_3mul_4/y*
_output_shapes
: *
T0
P
Greater_1/yConst*
_output_shapes
: *
valueB
 *���?*
dtype0
T
	Greater_1GreaterExpGreater_1/y*#
_output_shapes
:���������*
T0
K
Less/yConst*
dtype0*
valueB
 *��L?*
_output_shapes
: 
G
LessLessExpLess/y*#
_output_shapes
:���������*
T0
L
	LogicalOr	LogicalOr	Greater_1Less*#
_output_shapes
:���������
d
CastCast	LogicalOr*#
_output_shapes
:���������*
Truncate( *

DstT0*

SrcT0

Q
Const_4Const*
dtype0*
_output_shapes
:*
valueB: 
[
Mean_4MeanCastConst_4*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
L
mul_5/xConst*
dtype0*
_output_shapes
: *
valueB
 *���=
=
mul_5Mulmul_5/xmul_4*
_output_shapes
: *
T0
9
sub_3SubNegmul_5*
T0*
_output_shapes
: 
R
gradients/ShapeConst*
_output_shapes
: *
valueB *
dtype0
X
gradients/grad_ys_0Const*
valueB
 *  �?*
dtype0*
_output_shapes
: 
o
gradients/FillFillgradients/Shapegradients/grad_ys_0*
T0*
_output_shapes
: *

index_type0
P
gradients/sub_3_grad/NegNeggradients/Fill*
T0*
_output_shapes
: 
Y
%gradients/sub_3_grad/tuple/group_depsNoOp^gradients/Fill^gradients/sub_3_grad/Neg
�
-gradients/sub_3_grad/tuple/control_dependencyIdentitygradients/Fill&^gradients/sub_3_grad/tuple/group_deps*
_output_shapes
: *
T0*!
_class
loc:@gradients/Fill
�
/gradients/sub_3_grad/tuple/control_dependency_1Identitygradients/sub_3_grad/Neg&^gradients/sub_3_grad/tuple/group_deps*
_output_shapes
: *
T0*+
_class!
loc:@gradients/sub_3_grad/Neg
m
gradients/Neg_grad/NegNeg-gradients/sub_3_grad/tuple/control_dependency*
_output_shapes
: *
T0
x
gradients/mul_5_grad/MulMul/gradients/sub_3_grad/tuple/control_dependency_1mul_4*
_output_shapes
: *
T0
|
gradients/mul_5_grad/Mul_1Mul/gradients/sub_3_grad/tuple/control_dependency_1mul_5/x*
_output_shapes
: *
T0
e
%gradients/mul_5_grad/tuple/group_depsNoOp^gradients/mul_5_grad/Mul^gradients/mul_5_grad/Mul_1
�
-gradients/mul_5_grad/tuple/control_dependencyIdentitygradients/mul_5_grad/Mul&^gradients/mul_5_grad/tuple/group_deps*+
_class!
loc:@gradients/mul_5_grad/Mul*
T0*
_output_shapes
: 
�
/gradients/mul_5_grad/tuple/control_dependency_1Identitygradients/mul_5_grad/Mul_1&^gradients/mul_5_grad/tuple/group_deps*
_output_shapes
: *-
_class#
!loc:@gradients/mul_5_grad/Mul_1*
T0
k
!gradients/Mean_grad/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
�
gradients/Mean_grad/ReshapeReshapegradients/Neg_grad/Neg!gradients/Mean_grad/Reshape/shape*
_output_shapes
:*
T0*
Tshape0
`
gradients/Mean_grad/ShapeShapeMinimum*
out_type0*
T0*
_output_shapes
:
�
gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Shape*

Tmultiples0*
T0*#
_output_shapes
:���������
b
gradients/Mean_grad/Shape_1ShapeMinimum*
_output_shapes
:*
T0*
out_type0
^
gradients/Mean_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
c
gradients/Mean_grad/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
�
gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_1gradients/Mean_grad/Const*
T0*
	keep_dims( *
_output_shapes
: *

Tidx0
e
gradients/Mean_grad/Const_1Const*
dtype0*
_output_shapes
:*
valueB: 
�
gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_2gradients/Mean_grad/Const_1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
_
gradients/Mean_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
�
gradients/Mean_grad/MaximumMaximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum/y*
T0*
_output_shapes
: 
�
gradients/Mean_grad/floordivFloorDivgradients/Mean_grad/Prodgradients/Mean_grad/Maximum*
T0*
_output_shapes
: 
~
gradients/Mean_grad/CastCastgradients/Mean_grad/floordiv*

DstT0*
Truncate( *
_output_shapes
: *

SrcT0
�
gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*
T0*#
_output_shapes
:���������
z
gradients/mul_4_grad/MulMul/gradients/mul_5_grad/tuple/control_dependency_1mul_4/y*
_output_shapes
: *
T0
{
gradients/mul_4_grad/Mul_1Mul/gradients/mul_5_grad/tuple/control_dependency_1Mean_3*
_output_shapes
: *
T0
e
%gradients/mul_4_grad/tuple/group_depsNoOp^gradients/mul_4_grad/Mul^gradients/mul_4_grad/Mul_1
�
-gradients/mul_4_grad/tuple/control_dependencyIdentitygradients/mul_4_grad/Mul&^gradients/mul_4_grad/tuple/group_deps*+
_class!
loc:@gradients/mul_4_grad/Mul*
T0*
_output_shapes
: 
�
/gradients/mul_4_grad/tuple/control_dependency_1Identitygradients/mul_4_grad/Mul_1&^gradients/mul_4_grad/tuple/group_deps*
_output_shapes
: *
T0*-
_class#
!loc:@gradients/mul_4_grad/Mul_1
a
gradients/Minimum_grad/ShapeShapemul_2*
_output_shapes
:*
T0*
out_type0
d
gradients/Minimum_grad/Shape_1ShapeSelect*
T0*
out_type0*
_output_shapes
:
y
gradients/Minimum_grad/Shape_2Shapegradients/Mean_grad/truediv*
_output_shapes
:*
T0*
out_type0
g
"gradients/Minimum_grad/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
�
gradients/Minimum_grad/zerosFillgradients/Minimum_grad/Shape_2"gradients/Minimum_grad/zeros/Const*
T0*#
_output_shapes
:���������*

index_type0
j
 gradients/Minimum_grad/LessEqual	LessEqualmul_2Select*
T0*#
_output_shapes
:���������
�
,gradients/Minimum_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Minimum_grad/Shapegradients/Minimum_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
gradients/Minimum_grad/SelectSelect gradients/Minimum_grad/LessEqualgradients/Mean_grad/truedivgradients/Minimum_grad/zeros*#
_output_shapes
:���������*
T0
�
gradients/Minimum_grad/SumSumgradients/Minimum_grad/Select,gradients/Minimum_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
�
gradients/Minimum_grad/ReshapeReshapegradients/Minimum_grad/Sumgradients/Minimum_grad/Shape*
T0*
Tshape0*#
_output_shapes
:���������
�
gradients/Minimum_grad/Select_1Select gradients/Minimum_grad/LessEqualgradients/Minimum_grad/zerosgradients/Mean_grad/truediv*#
_output_shapes
:���������*
T0
�
gradients/Minimum_grad/Sum_1Sumgradients/Minimum_grad/Select_1.gradients/Minimum_grad/BroadcastGradientArgs:1*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
 gradients/Minimum_grad/Reshape_1Reshapegradients/Minimum_grad/Sum_1gradients/Minimum_grad/Shape_1*
T0*
Tshape0*#
_output_shapes
:���������
s
'gradients/Minimum_grad/tuple/group_depsNoOp^gradients/Minimum_grad/Reshape!^gradients/Minimum_grad/Reshape_1
�
/gradients/Minimum_grad/tuple/control_dependencyIdentitygradients/Minimum_grad/Reshape(^gradients/Minimum_grad/tuple/group_deps*1
_class'
%#loc:@gradients/Minimum_grad/Reshape*
T0*#
_output_shapes
:���������
�
1gradients/Minimum_grad/tuple/control_dependency_1Identity gradients/Minimum_grad/Reshape_1(^gradients/Minimum_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients/Minimum_grad/Reshape_1*#
_output_shapes
:���������
t
#gradients/Mean_3_grad/Reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
�
gradients/Mean_3_grad/ReshapeReshape-gradients/mul_4_grad/tuple/control_dependency#gradients/Mean_3_grad/Reshape/shape*
Tshape0*
_output_shapes

:*
T0
`
gradients/Mean_3_grad/ShapeShapemul_3*
out_type0*
T0*
_output_shapes
:
�
gradients/Mean_3_grad/TileTilegradients/Mean_3_grad/Reshapegradients/Mean_3_grad/Shape*

Tmultiples0*
T0*'
_output_shapes
:���������
b
gradients/Mean_3_grad/Shape_1Shapemul_3*
out_type0*
_output_shapes
:*
T0
`
gradients/Mean_3_grad/Shape_2Const*
dtype0*
valueB *
_output_shapes
: 
e
gradients/Mean_3_grad/ConstConst*
_output_shapes
:*
valueB: *
dtype0
�
gradients/Mean_3_grad/ProdProdgradients/Mean_3_grad/Shape_1gradients/Mean_3_grad/Const*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
g
gradients/Mean_3_grad/Const_1Const*
valueB: *
_output_shapes
:*
dtype0
�
gradients/Mean_3_grad/Prod_1Prodgradients/Mean_3_grad/Shape_2gradients/Mean_3_grad/Const_1*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
a
gradients/Mean_3_grad/Maximum/yConst*
_output_shapes
: *
value	B :*
dtype0
�
gradients/Mean_3_grad/MaximumMaximumgradients/Mean_3_grad/Prod_1gradients/Mean_3_grad/Maximum/y*
T0*
_output_shapes
: 
�
gradients/Mean_3_grad/floordivFloorDivgradients/Mean_3_grad/Prodgradients/Mean_3_grad/Maximum*
T0*
_output_shapes
: 
�
gradients/Mean_3_grad/CastCastgradients/Mean_3_grad/floordiv*

SrcT0*
_output_shapes
: *
Truncate( *

DstT0
�
gradients/Mean_3_grad/truedivRealDivgradients/Mean_3_grad/Tilegradients/Mean_3_grad/Cast*'
_output_shapes
:���������*
T0
]
gradients/mul_2_grad/ShapeShapeExp*
_output_shapes
:*
T0*
out_type0
i
gradients/mul_2_grad/Shape_1ShapePlaceholder_2*
out_type0*
_output_shapes
:*
T0
�
*gradients/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/mul_2_grad/Shapegradients/mul_2_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
gradients/mul_2_grad/MulMul/gradients/Minimum_grad/tuple/control_dependencyPlaceholder_2*#
_output_shapes
:���������*
T0
�
gradients/mul_2_grad/SumSumgradients/mul_2_grad/Mul*gradients/mul_2_grad/BroadcastGradientArgs*

Tidx0*
_output_shapes
:*
T0*
	keep_dims( 
�
gradients/mul_2_grad/ReshapeReshapegradients/mul_2_grad/Sumgradients/mul_2_grad/Shape*
Tshape0*
T0*#
_output_shapes
:���������
�
gradients/mul_2_grad/Mul_1MulExp/gradients/Minimum_grad/tuple/control_dependency*#
_output_shapes
:���������*
T0
�
gradients/mul_2_grad/Sum_1Sumgradients/mul_2_grad/Mul_1,gradients/mul_2_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
T0*
	keep_dims( 
�
gradients/mul_2_grad/Reshape_1Reshapegradients/mul_2_grad/Sum_1gradients/mul_2_grad/Shape_1*
Tshape0*#
_output_shapes
:���������*
T0
m
%gradients/mul_2_grad/tuple/group_depsNoOp^gradients/mul_2_grad/Reshape^gradients/mul_2_grad/Reshape_1
�
-gradients/mul_2_grad/tuple/control_dependencyIdentitygradients/mul_2_grad/Reshape&^gradients/mul_2_grad/tuple/group_deps*
T0*#
_output_shapes
:���������*/
_class%
#!loc:@gradients/mul_2_grad/Reshape
�
/gradients/mul_2_grad/tuple/control_dependency_1Identitygradients/mul_2_grad/Reshape_1&^gradients/mul_2_grad/tuple/group_deps*#
_output_shapes
:���������*1
_class'
%#loc:@gradients/mul_2_grad/Reshape_1*
T0
_
gradients/mul_3_grad/ShapeShapeNeg_1*
out_type0*
T0*
_output_shapes
:
a
gradients/mul_3_grad/Shape_1ShapeExp_1*
_output_shapes
:*
T0*
out_type0
�
*gradients/mul_3_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/mul_3_grad/Shapegradients/mul_3_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
w
gradients/mul_3_grad/MulMulgradients/Mean_3_grad/truedivExp_1*
T0*'
_output_shapes
:���������
�
gradients/mul_3_grad/SumSumgradients/mul_3_grad/Mul*gradients/mul_3_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
�
gradients/mul_3_grad/ReshapeReshapegradients/mul_3_grad/Sumgradients/mul_3_grad/Shape*
T0*'
_output_shapes
:���������*
Tshape0
y
gradients/mul_3_grad/Mul_1MulNeg_1gradients/Mean_3_grad/truediv*
T0*'
_output_shapes
:���������
�
gradients/mul_3_grad/Sum_1Sumgradients/mul_3_grad/Mul_1,gradients/mul_3_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
_output_shapes
:*
T0
�
gradients/mul_3_grad/Reshape_1Reshapegradients/mul_3_grad/Sum_1gradients/mul_3_grad/Shape_1*
T0*'
_output_shapes
:���������*
Tshape0
m
%gradients/mul_3_grad/tuple/group_depsNoOp^gradients/mul_3_grad/Reshape^gradients/mul_3_grad/Reshape_1
�
-gradients/mul_3_grad/tuple/control_dependencyIdentitygradients/mul_3_grad/Reshape&^gradients/mul_3_grad/tuple/group_deps*'
_output_shapes
:���������*/
_class%
#!loc:@gradients/mul_3_grad/Reshape*
T0
�
/gradients/mul_3_grad/tuple/control_dependency_1Identitygradients/mul_3_grad/Reshape_1&^gradients/mul_3_grad/tuple/group_deps*1
_class'
%#loc:@gradients/mul_3_grad/Reshape_1*
T0*'
_output_shapes
:���������

gradients/Exp_grad/mulMul-gradients/mul_2_grad/tuple/control_dependencyExp*
T0*#
_output_shapes
:���������
�
gradients/Neg_1_grad/NegNeg-gradients/mul_3_grad/tuple/control_dependency*'
_output_shapes
:���������*
T0
�
gradients/Exp_1_grad/mulMul/gradients/mul_3_grad/tuple/control_dependency_1Exp_1*
T0*'
_output_shapes
:���������
^
gradients/sub_grad/ShapeShapepi/Sum*
out_type0*
_output_shapes
:*
T0
g
gradients/sub_grad/Shape_1ShapePlaceholder_4*
out_type0*
T0*
_output_shapes
:
�
(gradients/sub_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/sub_grad/Shapegradients/sub_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/sub_grad/SumSumgradients/Exp_grad/mul(gradients/sub_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
gradients/sub_grad/ReshapeReshapegradients/sub_grad/Sumgradients/sub_grad/Shape*
Tshape0*
T0*#
_output_shapes
:���������
�
gradients/sub_grad/Sum_1Sumgradients/Exp_grad/mul*gradients/sub_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
Z
gradients/sub_grad/NegNeggradients/sub_grad/Sum_1*
_output_shapes
:*
T0
�
gradients/sub_grad/Reshape_1Reshapegradients/sub_grad/Neggradients/sub_grad/Shape_1*#
_output_shapes
:���������*
T0*
Tshape0
g
#gradients/sub_grad/tuple/group_depsNoOp^gradients/sub_grad/Reshape^gradients/sub_grad/Reshape_1
�
+gradients/sub_grad/tuple/control_dependencyIdentitygradients/sub_grad/Reshape$^gradients/sub_grad/tuple/group_deps*
T0*#
_output_shapes
:���������*-
_class#
!loc:@gradients/sub_grad/Reshape
�
-gradients/sub_grad/tuple/control_dependency_1Identitygradients/sub_grad/Reshape_1$^gradients/sub_grad/tuple/group_deps*
T0*#
_output_shapes
:���������*/
_class%
#!loc:@gradients/sub_grad/Reshape_1
a
gradients/pi/Sum_grad/ShapeShapepi/mul*
out_type0*
T0*
_output_shapes
:
�
gradients/pi/Sum_grad/SizeConst*
dtype0*
_output_shapes
: *
value	B :*.
_class$
" loc:@gradients/pi/Sum_grad/Shape
�
gradients/pi/Sum_grad/addAddpi/Sum/reduction_indicesgradients/pi/Sum_grad/Size*
_output_shapes
: *
T0*.
_class$
" loc:@gradients/pi/Sum_grad/Shape
�
gradients/pi/Sum_grad/modFloorModgradients/pi/Sum_grad/addgradients/pi/Sum_grad/Size*.
_class$
" loc:@gradients/pi/Sum_grad/Shape*
_output_shapes
: *
T0
�
gradients/pi/Sum_grad/Shape_1Const*
_output_shapes
: *.
_class$
" loc:@gradients/pi/Sum_grad/Shape*
valueB *
dtype0
�
!gradients/pi/Sum_grad/range/startConst*
dtype0*
_output_shapes
: *.
_class$
" loc:@gradients/pi/Sum_grad/Shape*
value	B : 
�
!gradients/pi/Sum_grad/range/deltaConst*
value	B :*
dtype0*.
_class$
" loc:@gradients/pi/Sum_grad/Shape*
_output_shapes
: 
�
gradients/pi/Sum_grad/rangeRange!gradients/pi/Sum_grad/range/startgradients/pi/Sum_grad/Size!gradients/pi/Sum_grad/range/delta*

Tidx0*.
_class$
" loc:@gradients/pi/Sum_grad/Shape*
_output_shapes
:
�
 gradients/pi/Sum_grad/Fill/valueConst*
value	B :*.
_class$
" loc:@gradients/pi/Sum_grad/Shape*
_output_shapes
: *
dtype0
�
gradients/pi/Sum_grad/FillFillgradients/pi/Sum_grad/Shape_1 gradients/pi/Sum_grad/Fill/value*
T0*
_output_shapes
: *.
_class$
" loc:@gradients/pi/Sum_grad/Shape*

index_type0
�
#gradients/pi/Sum_grad/DynamicStitchDynamicStitchgradients/pi/Sum_grad/rangegradients/pi/Sum_grad/modgradients/pi/Sum_grad/Shapegradients/pi/Sum_grad/Fill*
T0*
_output_shapes
:*.
_class$
" loc:@gradients/pi/Sum_grad/Shape*
N
�
gradients/pi/Sum_grad/Maximum/yConst*
dtype0*
_output_shapes
: *
value	B :*.
_class$
" loc:@gradients/pi/Sum_grad/Shape
�
gradients/pi/Sum_grad/MaximumMaximum#gradients/pi/Sum_grad/DynamicStitchgradients/pi/Sum_grad/Maximum/y*
T0*
_output_shapes
:*.
_class$
" loc:@gradients/pi/Sum_grad/Shape
�
gradients/pi/Sum_grad/floordivFloorDivgradients/pi/Sum_grad/Shapegradients/pi/Sum_grad/Maximum*
T0*
_output_shapes
:*.
_class$
" loc:@gradients/pi/Sum_grad/Shape
�
gradients/pi/Sum_grad/ReshapeReshape+gradients/sub_grad/tuple/control_dependency#gradients/pi/Sum_grad/DynamicStitch*
T0*0
_output_shapes
:������������������*
Tshape0
�
gradients/pi/Sum_grad/TileTilegradients/pi/Sum_grad/Reshapegradients/pi/Sum_grad/floordiv*'
_output_shapes
:���������*

Tmultiples0*
T0
e
gradients/pi/mul_grad/ShapeShape
pi/one_hot*
T0*
_output_shapes
:*
out_type0
j
gradients/pi/mul_grad/Shape_1Shapepi/LogSoftmax*
out_type0*
T0*
_output_shapes
:
�
+gradients/pi/mul_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/pi/mul_grad/Shapegradients/pi/mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
}
gradients/pi/mul_grad/MulMulgradients/pi/Sum_grad/Tilepi/LogSoftmax*'
_output_shapes
:���������*
T0
�
gradients/pi/mul_grad/SumSumgradients/pi/mul_grad/Mul+gradients/pi/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
gradients/pi/mul_grad/ReshapeReshapegradients/pi/mul_grad/Sumgradients/pi/mul_grad/Shape*
Tshape0*'
_output_shapes
:���������*
T0
|
gradients/pi/mul_grad/Mul_1Mul
pi/one_hotgradients/pi/Sum_grad/Tile*'
_output_shapes
:���������*
T0
�
gradients/pi/mul_grad/Sum_1Sumgradients/pi/mul_grad/Mul_1-gradients/pi/mul_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
gradients/pi/mul_grad/Reshape_1Reshapegradients/pi/mul_grad/Sum_1gradients/pi/mul_grad/Shape_1*'
_output_shapes
:���������*
T0*
Tshape0
p
&gradients/pi/mul_grad/tuple/group_depsNoOp^gradients/pi/mul_grad/Reshape ^gradients/pi/mul_grad/Reshape_1
�
.gradients/pi/mul_grad/tuple/control_dependencyIdentitygradients/pi/mul_grad/Reshape'^gradients/pi/mul_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*0
_class&
$"loc:@gradients/pi/mul_grad/Reshape
�
0gradients/pi/mul_grad/tuple/control_dependency_1Identitygradients/pi/mul_grad/Reshape_1'^gradients/pi/mul_grad/tuple/group_deps*'
_output_shapes
:���������*2
_class(
&$loc:@gradients/pi/mul_grad/Reshape_1*
T0
�
gradients/AddNAddNgradients/Neg_1_grad/Neggradients/Exp_1_grad/mul0gradients/pi/mul_grad/tuple/control_dependency_1*
N*'
_output_shapes
:���������*
T0*+
_class!
loc:@gradients/Neg_1_grad/Neg
h
 gradients/pi/LogSoftmax_grad/ExpExppi/LogSoftmax*'
_output_shapes
:���������*
T0
}
2gradients/pi/LogSoftmax_grad/Sum/reduction_indicesConst*
_output_shapes
: *
valueB :
���������*
dtype0
�
 gradients/pi/LogSoftmax_grad/SumSumgradients/AddN2gradients/pi/LogSoftmax_grad/Sum/reduction_indices*
	keep_dims(*'
_output_shapes
:���������*

Tidx0*
T0
�
 gradients/pi/LogSoftmax_grad/mulMul gradients/pi/LogSoftmax_grad/Sum gradients/pi/LogSoftmax_grad/Exp*'
_output_shapes
:���������*
T0
�
 gradients/pi/LogSoftmax_grad/subSubgradients/AddN gradients/pi/LogSoftmax_grad/mul*'
_output_shapes
:���������*
T0
�
-gradients/pi/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad gradients/pi/LogSoftmax_grad/sub*
_output_shapes
:*
data_formatNHWC*
T0
�
2gradients/pi/dense_2/BiasAdd_grad/tuple/group_depsNoOp!^gradients/pi/LogSoftmax_grad/sub.^gradients/pi/dense_2/BiasAdd_grad/BiasAddGrad
�
:gradients/pi/dense_2/BiasAdd_grad/tuple/control_dependencyIdentity gradients/pi/LogSoftmax_grad/sub3^gradients/pi/dense_2/BiasAdd_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients/pi/LogSoftmax_grad/sub*'
_output_shapes
:���������
�
<gradients/pi/dense_2/BiasAdd_grad/tuple/control_dependency_1Identity-gradients/pi/dense_2/BiasAdd_grad/BiasAddGrad3^gradients/pi/dense_2/BiasAdd_grad/tuple/group_deps*
T0*
_output_shapes
:*@
_class6
42loc:@gradients/pi/dense_2/BiasAdd_grad/BiasAddGrad
�
'gradients/pi/dense_2/MatMul_grad/MatMulMatMul:gradients/pi/dense_2/BiasAdd_grad/tuple/control_dependencypi/dense_2/kernel/read*
transpose_b(*
transpose_a( *'
_output_shapes
:���������@*
T0
�
)gradients/pi/dense_2/MatMul_grad/MatMul_1MatMulpi/dense_1/Tanh:gradients/pi/dense_2/BiasAdd_grad/tuple/control_dependency*
transpose_a(*
transpose_b( *
T0*
_output_shapes

:@
�
1gradients/pi/dense_2/MatMul_grad/tuple/group_depsNoOp(^gradients/pi/dense_2/MatMul_grad/MatMul*^gradients/pi/dense_2/MatMul_grad/MatMul_1
�
9gradients/pi/dense_2/MatMul_grad/tuple/control_dependencyIdentity'gradients/pi/dense_2/MatMul_grad/MatMul2^gradients/pi/dense_2/MatMul_grad/tuple/group_deps*:
_class0
.,loc:@gradients/pi/dense_2/MatMul_grad/MatMul*
T0*'
_output_shapes
:���������@
�
;gradients/pi/dense_2/MatMul_grad/tuple/control_dependency_1Identity)gradients/pi/dense_2/MatMul_grad/MatMul_12^gradients/pi/dense_2/MatMul_grad/tuple/group_deps*
_output_shapes

:@*<
_class2
0.loc:@gradients/pi/dense_2/MatMul_grad/MatMul_1*
T0
�
'gradients/pi/dense_1/Tanh_grad/TanhGradTanhGradpi/dense_1/Tanh9gradients/pi/dense_2/MatMul_grad/tuple/control_dependency*'
_output_shapes
:���������@*
T0
�
-gradients/pi/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad'gradients/pi/dense_1/Tanh_grad/TanhGrad*
T0*
_output_shapes
:@*
data_formatNHWC
�
2gradients/pi/dense_1/BiasAdd_grad/tuple/group_depsNoOp.^gradients/pi/dense_1/BiasAdd_grad/BiasAddGrad(^gradients/pi/dense_1/Tanh_grad/TanhGrad
�
:gradients/pi/dense_1/BiasAdd_grad/tuple/control_dependencyIdentity'gradients/pi/dense_1/Tanh_grad/TanhGrad3^gradients/pi/dense_1/BiasAdd_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/pi/dense_1/Tanh_grad/TanhGrad*'
_output_shapes
:���������@
�
<gradients/pi/dense_1/BiasAdd_grad/tuple/control_dependency_1Identity-gradients/pi/dense_1/BiasAdd_grad/BiasAddGrad3^gradients/pi/dense_1/BiasAdd_grad/tuple/group_deps*
_output_shapes
:@*@
_class6
42loc:@gradients/pi/dense_1/BiasAdd_grad/BiasAddGrad*
T0
�
'gradients/pi/dense_1/MatMul_grad/MatMulMatMul:gradients/pi/dense_1/BiasAdd_grad/tuple/control_dependencypi/dense_1/kernel/read*
transpose_b(*
transpose_a( *
T0*'
_output_shapes
:���������@
�
)gradients/pi/dense_1/MatMul_grad/MatMul_1MatMulpi/dense/Tanh:gradients/pi/dense_1/BiasAdd_grad/tuple/control_dependency*
transpose_a(*
T0*
transpose_b( *
_output_shapes

:@@
�
1gradients/pi/dense_1/MatMul_grad/tuple/group_depsNoOp(^gradients/pi/dense_1/MatMul_grad/MatMul*^gradients/pi/dense_1/MatMul_grad/MatMul_1
�
9gradients/pi/dense_1/MatMul_grad/tuple/control_dependencyIdentity'gradients/pi/dense_1/MatMul_grad/MatMul2^gradients/pi/dense_1/MatMul_grad/tuple/group_deps*'
_output_shapes
:���������@*:
_class0
.,loc:@gradients/pi/dense_1/MatMul_grad/MatMul*
T0
�
;gradients/pi/dense_1/MatMul_grad/tuple/control_dependency_1Identity)gradients/pi/dense_1/MatMul_grad/MatMul_12^gradients/pi/dense_1/MatMul_grad/tuple/group_deps*<
_class2
0.loc:@gradients/pi/dense_1/MatMul_grad/MatMul_1*
_output_shapes

:@@*
T0
�
%gradients/pi/dense/Tanh_grad/TanhGradTanhGradpi/dense/Tanh9gradients/pi/dense_1/MatMul_grad/tuple/control_dependency*'
_output_shapes
:���������@*
T0
�
+gradients/pi/dense/BiasAdd_grad/BiasAddGradBiasAddGrad%gradients/pi/dense/Tanh_grad/TanhGrad*
T0*
data_formatNHWC*
_output_shapes
:@
�
0gradients/pi/dense/BiasAdd_grad/tuple/group_depsNoOp,^gradients/pi/dense/BiasAdd_grad/BiasAddGrad&^gradients/pi/dense/Tanh_grad/TanhGrad
�
8gradients/pi/dense/BiasAdd_grad/tuple/control_dependencyIdentity%gradients/pi/dense/Tanh_grad/TanhGrad1^gradients/pi/dense/BiasAdd_grad/tuple/group_deps*8
_class.
,*loc:@gradients/pi/dense/Tanh_grad/TanhGrad*'
_output_shapes
:���������@*
T0
�
:gradients/pi/dense/BiasAdd_grad/tuple/control_dependency_1Identity+gradients/pi/dense/BiasAdd_grad/BiasAddGrad1^gradients/pi/dense/BiasAdd_grad/tuple/group_deps*
_output_shapes
:@*
T0*>
_class4
20loc:@gradients/pi/dense/BiasAdd_grad/BiasAddGrad
�
%gradients/pi/dense/MatMul_grad/MatMulMatMul8gradients/pi/dense/BiasAdd_grad/tuple/control_dependencypi/dense/kernel/read*
transpose_b(*'
_output_shapes
:���������*
T0*
transpose_a( 
�
'gradients/pi/dense/MatMul_grad/MatMul_1MatMulPlaceholder8gradients/pi/dense/BiasAdd_grad/tuple/control_dependency*
_output_shapes

:@*
T0*
transpose_b( *
transpose_a(
�
/gradients/pi/dense/MatMul_grad/tuple/group_depsNoOp&^gradients/pi/dense/MatMul_grad/MatMul(^gradients/pi/dense/MatMul_grad/MatMul_1
�
7gradients/pi/dense/MatMul_grad/tuple/control_dependencyIdentity%gradients/pi/dense/MatMul_grad/MatMul0^gradients/pi/dense/MatMul_grad/tuple/group_deps*
T0*8
_class.
,*loc:@gradients/pi/dense/MatMul_grad/MatMul*'
_output_shapes
:���������
�
9gradients/pi/dense/MatMul_grad/tuple/control_dependency_1Identity'gradients/pi/dense/MatMul_grad/MatMul_10^gradients/pi/dense/MatMul_grad/tuple/group_deps*
T0*
_output_shapes

:@*:
_class0
.,loc:@gradients/pi/dense/MatMul_grad/MatMul_1
`
Reshape/shapeConst*
_output_shapes
:*
valueB:
���������*
dtype0
�
ReshapeReshape9gradients/pi/dense/MatMul_grad/tuple/control_dependency_1Reshape/shape*
Tshape0*
_output_shapes	
:�*
T0
b
Reshape_1/shapeConst*
_output_shapes
:*
valueB:
���������*
dtype0
�
	Reshape_1Reshape:gradients/pi/dense/BiasAdd_grad/tuple/control_dependency_1Reshape_1/shape*
T0*
Tshape0*
_output_shapes
:@
b
Reshape_2/shapeConst*
_output_shapes
:*
valueB:
���������*
dtype0
�
	Reshape_2Reshape;gradients/pi/dense_1/MatMul_grad/tuple/control_dependency_1Reshape_2/shape*
T0*
Tshape0*
_output_shapes	
:� 
b
Reshape_3/shapeConst*
dtype0*
_output_shapes
:*
valueB:
���������
�
	Reshape_3Reshape<gradients/pi/dense_1/BiasAdd_grad/tuple/control_dependency_1Reshape_3/shape*
Tshape0*
T0*
_output_shapes
:@
b
Reshape_4/shapeConst*
valueB:
���������*
_output_shapes
:*
dtype0
�
	Reshape_4Reshape;gradients/pi/dense_2/MatMul_grad/tuple/control_dependency_1Reshape_4/shape*
T0*
_output_shapes	
:�*
Tshape0
b
Reshape_5/shapeConst*
dtype0*
valueB:
���������*
_output_shapes
:
�
	Reshape_5Reshape<gradients/pi/dense_2/BiasAdd_grad/tuple/control_dependency_1Reshape_5/shape*
T0*
Tshape0*
_output_shapes
:
M
concat/axisConst*
_output_shapes
: *
value	B : *
dtype0
�
concatConcatV2Reshape	Reshape_1	Reshape_2	Reshape_3	Reshape_4	Reshape_5concat/axis*
_output_shapes	
:�%*
N*
T0*

Tidx0
g
PyFuncPyFuncconcat*
Tin
2*
Tout
2*
token
pyfunc_0*
_output_shapes	
:�%
h
Const_5Const*-
value$B""�  @      @   �      *
_output_shapes
:*
dtype0
Q
split/split_dimConst*
_output_shapes
: *
value	B : *
dtype0
�
splitSplitVPyFuncConst_5split/split_dim*

Tlen0*
	num_split*
T0*;
_output_shapes)
':�:@:� :@:�:
`
Reshape_6/shapeConst*
valueB"   @   *
dtype0*
_output_shapes
:
c
	Reshape_6ReshapesplitReshape_6/shape*
T0*
_output_shapes

:@*
Tshape0
Y
Reshape_7/shapeConst*
valueB:@*
_output_shapes
:*
dtype0
a
	Reshape_7Reshapesplit:1Reshape_7/shape*
T0*
_output_shapes
:@*
Tshape0
`
Reshape_8/shapeConst*
dtype0*
_output_shapes
:*
valueB"@   @   
e
	Reshape_8Reshapesplit:2Reshape_8/shape*
T0*
Tshape0*
_output_shapes

:@@
Y
Reshape_9/shapeConst*
valueB:@*
_output_shapes
:*
dtype0
a
	Reshape_9Reshapesplit:3Reshape_9/shape*
T0*
_output_shapes
:@*
Tshape0
a
Reshape_10/shapeConst*
valueB"@      *
dtype0*
_output_shapes
:
g

Reshape_10Reshapesplit:4Reshape_10/shape*
_output_shapes

:@*
T0*
Tshape0
Z
Reshape_11/shapeConst*
valueB:*
_output_shapes
:*
dtype0
c

Reshape_11Reshapesplit:5Reshape_11/shape*
Tshape0*
_output_shapes
:*
T0
�
beta1_power/initial_valueConst* 
_class
loc:@pi/dense/bias*
dtype0*
_output_shapes
: *
valueB
 *fff?
�
beta1_power
VariableV2*
	container *
shape: *
dtype0* 
_class
loc:@pi/dense/bias*
_output_shapes
: *
shared_name 
�
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value* 
_class
loc:@pi/dense/bias*
T0*
use_locking(*
validate_shape(*
_output_shapes
: 
l
beta1_power/readIdentitybeta1_power*
_output_shapes
: *
T0* 
_class
loc:@pi/dense/bias
�
beta2_power/initial_valueConst*
valueB
 *w�?*
dtype0* 
_class
loc:@pi/dense/bias*
_output_shapes
: 
�
beta2_power
VariableV2*
	container * 
_class
loc:@pi/dense/bias*
dtype0*
shared_name *
_output_shapes
: *
shape: 
�
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
T0*
_output_shapes
: *
validate_shape(* 
_class
loc:@pi/dense/bias*
use_locking(
l
beta2_power/readIdentitybeta2_power*
T0* 
_class
loc:@pi/dense/bias*
_output_shapes
: 
�
&pi/dense/kernel/Adam/Initializer/zerosConst*"
_class
loc:@pi/dense/kernel*
valueB@*    *
dtype0*
_output_shapes

:@
�
pi/dense/kernel/Adam
VariableV2*
shape
:@*"
_class
loc:@pi/dense/kernel*
shared_name *
_output_shapes

:@*
	container *
dtype0
�
pi/dense/kernel/Adam/AssignAssignpi/dense/kernel/Adam&pi/dense/kernel/Adam/Initializer/zeros*
_output_shapes

:@*
validate_shape(*
T0*
use_locking(*"
_class
loc:@pi/dense/kernel
�
pi/dense/kernel/Adam/readIdentitypi/dense/kernel/Adam*
_output_shapes

:@*"
_class
loc:@pi/dense/kernel*
T0
�
(pi/dense/kernel/Adam_1/Initializer/zerosConst*
dtype0*"
_class
loc:@pi/dense/kernel*
valueB@*    *
_output_shapes

:@
�
pi/dense/kernel/Adam_1
VariableV2*
	container *
dtype0*
shape
:@*"
_class
loc:@pi/dense/kernel*
shared_name *
_output_shapes

:@
�
pi/dense/kernel/Adam_1/AssignAssignpi/dense/kernel/Adam_1(pi/dense/kernel/Adam_1/Initializer/zeros*
validate_shape(*
use_locking(*"
_class
loc:@pi/dense/kernel*
_output_shapes

:@*
T0
�
pi/dense/kernel/Adam_1/readIdentitypi/dense/kernel/Adam_1*"
_class
loc:@pi/dense/kernel*
T0*
_output_shapes

:@
�
$pi/dense/bias/Adam/Initializer/zerosConst* 
_class
loc:@pi/dense/bias*
valueB@*    *
_output_shapes
:@*
dtype0
�
pi/dense/bias/Adam
VariableV2*
_output_shapes
:@*
shared_name *
shape:@*
	container *
dtype0* 
_class
loc:@pi/dense/bias
�
pi/dense/bias/Adam/AssignAssignpi/dense/bias/Adam$pi/dense/bias/Adam/Initializer/zeros*
validate_shape(*
T0*
_output_shapes
:@* 
_class
loc:@pi/dense/bias*
use_locking(
~
pi/dense/bias/Adam/readIdentitypi/dense/bias/Adam*
T0*
_output_shapes
:@* 
_class
loc:@pi/dense/bias
�
&pi/dense/bias/Adam_1/Initializer/zerosConst*
dtype0* 
_class
loc:@pi/dense/bias*
_output_shapes
:@*
valueB@*    
�
pi/dense/bias/Adam_1
VariableV2*
	container *
dtype0*
shared_name * 
_class
loc:@pi/dense/bias*
_output_shapes
:@*
shape:@
�
pi/dense/bias/Adam_1/AssignAssignpi/dense/bias/Adam_1&pi/dense/bias/Adam_1/Initializer/zeros*
T0*
validate_shape(*
use_locking(* 
_class
loc:@pi/dense/bias*
_output_shapes
:@
�
pi/dense/bias/Adam_1/readIdentitypi/dense/bias/Adam_1*
T0* 
_class
loc:@pi/dense/bias*
_output_shapes
:@
�
8pi/dense_1/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
valueB"@   @   *
_output_shapes
:*
dtype0*$
_class
loc:@pi/dense_1/kernel
�
.pi/dense_1/kernel/Adam/Initializer/zeros/ConstConst*
_output_shapes
: *$
_class
loc:@pi/dense_1/kernel*
dtype0*
valueB
 *    
�
(pi/dense_1/kernel/Adam/Initializer/zerosFill8pi/dense_1/kernel/Adam/Initializer/zeros/shape_as_tensor.pi/dense_1/kernel/Adam/Initializer/zeros/Const*
_output_shapes

:@@*
T0*$
_class
loc:@pi/dense_1/kernel*

index_type0
�
pi/dense_1/kernel/Adam
VariableV2*
shared_name *
_output_shapes

:@@*
dtype0*
shape
:@@*
	container *$
_class
loc:@pi/dense_1/kernel
�
pi/dense_1/kernel/Adam/AssignAssignpi/dense_1/kernel/Adam(pi/dense_1/kernel/Adam/Initializer/zeros*
T0*
validate_shape(*
use_locking(*
_output_shapes

:@@*$
_class
loc:@pi/dense_1/kernel
�
pi/dense_1/kernel/Adam/readIdentitypi/dense_1/kernel/Adam*$
_class
loc:@pi/dense_1/kernel*
T0*
_output_shapes

:@@
�
:pi/dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*$
_class
loc:@pi/dense_1/kernel*
_output_shapes
:*
valueB"@   @   *
dtype0
�
0pi/dense_1/kernel/Adam_1/Initializer/zeros/ConstConst*$
_class
loc:@pi/dense_1/kernel*
_output_shapes
: *
valueB
 *    *
dtype0
�
*pi/dense_1/kernel/Adam_1/Initializer/zerosFill:pi/dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensor0pi/dense_1/kernel/Adam_1/Initializer/zeros/Const*

index_type0*
T0*$
_class
loc:@pi/dense_1/kernel*
_output_shapes

:@@
�
pi/dense_1/kernel/Adam_1
VariableV2*
	container *
_output_shapes

:@@*$
_class
loc:@pi/dense_1/kernel*
dtype0*
shared_name *
shape
:@@
�
pi/dense_1/kernel/Adam_1/AssignAssignpi/dense_1/kernel/Adam_1*pi/dense_1/kernel/Adam_1/Initializer/zeros*
T0*
validate_shape(*
use_locking(*
_output_shapes

:@@*$
_class
loc:@pi/dense_1/kernel
�
pi/dense_1/kernel/Adam_1/readIdentitypi/dense_1/kernel/Adam_1*$
_class
loc:@pi/dense_1/kernel*
_output_shapes

:@@*
T0
�
&pi/dense_1/bias/Adam/Initializer/zerosConst*
valueB@*    *
_output_shapes
:@*"
_class
loc:@pi/dense_1/bias*
dtype0
�
pi/dense_1/bias/Adam
VariableV2*
	container *"
_class
loc:@pi/dense_1/bias*
shared_name *
dtype0*
shape:@*
_output_shapes
:@
�
pi/dense_1/bias/Adam/AssignAssignpi/dense_1/bias/Adam&pi/dense_1/bias/Adam/Initializer/zeros*
use_locking(*
_output_shapes
:@*
T0*
validate_shape(*"
_class
loc:@pi/dense_1/bias
�
pi/dense_1/bias/Adam/readIdentitypi/dense_1/bias/Adam*
T0*"
_class
loc:@pi/dense_1/bias*
_output_shapes
:@
�
(pi/dense_1/bias/Adam_1/Initializer/zerosConst*
_output_shapes
:@*
valueB@*    *
dtype0*"
_class
loc:@pi/dense_1/bias
�
pi/dense_1/bias/Adam_1
VariableV2*
	container *
_output_shapes
:@*"
_class
loc:@pi/dense_1/bias*
shared_name *
dtype0*
shape:@
�
pi/dense_1/bias/Adam_1/AssignAssignpi/dense_1/bias/Adam_1(pi/dense_1/bias/Adam_1/Initializer/zeros*
validate_shape(*
T0*"
_class
loc:@pi/dense_1/bias*
_output_shapes
:@*
use_locking(
�
pi/dense_1/bias/Adam_1/readIdentitypi/dense_1/bias/Adam_1*
T0*"
_class
loc:@pi/dense_1/bias*
_output_shapes
:@
�
(pi/dense_2/kernel/Adam/Initializer/zerosConst*$
_class
loc:@pi/dense_2/kernel*
valueB@*    *
dtype0*
_output_shapes

:@
�
pi/dense_2/kernel/Adam
VariableV2*
dtype0*
	container *$
_class
loc:@pi/dense_2/kernel*
_output_shapes

:@*
shape
:@*
shared_name 
�
pi/dense_2/kernel/Adam/AssignAssignpi/dense_2/kernel/Adam(pi/dense_2/kernel/Adam/Initializer/zeros*$
_class
loc:@pi/dense_2/kernel*
T0*
_output_shapes

:@*
use_locking(*
validate_shape(
�
pi/dense_2/kernel/Adam/readIdentitypi/dense_2/kernel/Adam*
T0*
_output_shapes

:@*$
_class
loc:@pi/dense_2/kernel
�
*pi/dense_2/kernel/Adam_1/Initializer/zerosConst*
_output_shapes

:@*
valueB@*    *$
_class
loc:@pi/dense_2/kernel*
dtype0
�
pi/dense_2/kernel/Adam_1
VariableV2*
dtype0*
_output_shapes

:@*
	container *$
_class
loc:@pi/dense_2/kernel*
shape
:@*
shared_name 
�
pi/dense_2/kernel/Adam_1/AssignAssignpi/dense_2/kernel/Adam_1*pi/dense_2/kernel/Adam_1/Initializer/zeros*
T0*
validate_shape(*$
_class
loc:@pi/dense_2/kernel*
use_locking(*
_output_shapes

:@
�
pi/dense_2/kernel/Adam_1/readIdentitypi/dense_2/kernel/Adam_1*$
_class
loc:@pi/dense_2/kernel*
T0*
_output_shapes

:@
�
&pi/dense_2/bias/Adam/Initializer/zerosConst*
_output_shapes
:*
valueB*    *
dtype0*"
_class
loc:@pi/dense_2/bias
�
pi/dense_2/bias/Adam
VariableV2*
shape:*
_output_shapes
:*
	container *
shared_name *"
_class
loc:@pi/dense_2/bias*
dtype0
�
pi/dense_2/bias/Adam/AssignAssignpi/dense_2/bias/Adam&pi/dense_2/bias/Adam/Initializer/zeros*
_output_shapes
:*
use_locking(*"
_class
loc:@pi/dense_2/bias*
T0*
validate_shape(
�
pi/dense_2/bias/Adam/readIdentitypi/dense_2/bias/Adam*
_output_shapes
:*
T0*"
_class
loc:@pi/dense_2/bias
�
(pi/dense_2/bias/Adam_1/Initializer/zerosConst*
valueB*    *"
_class
loc:@pi/dense_2/bias*
dtype0*
_output_shapes
:
�
pi/dense_2/bias/Adam_1
VariableV2*
shared_name *
shape:*
dtype0*
	container *
_output_shapes
:*"
_class
loc:@pi/dense_2/bias
�
pi/dense_2/bias/Adam_1/AssignAssignpi/dense_2/bias/Adam_1(pi/dense_2/bias/Adam_1/Initializer/zeros*
_output_shapes
:*
T0*
use_locking(*
validate_shape(*"
_class
loc:@pi/dense_2/bias
�
pi/dense_2/bias/Adam_1/readIdentitypi/dense_2/bias/Adam_1*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:*
T0
W
Adam/learning_rateConst*
valueB
 *RI�9*
dtype0*
_output_shapes
: 
O

Adam/beta1Const*
_output_shapes
: *
dtype0*
valueB
 *fff?
O

Adam/beta2Const*
dtype0*
_output_shapes
: *
valueB
 *w�?
Q
Adam/epsilonConst*
valueB
 *w�+2*
dtype0*
_output_shapes
: 
�
%Adam/update_pi/dense/kernel/ApplyAdam	ApplyAdampi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon	Reshape_6*
use_nesterov( *
use_locking( *"
_class
loc:@pi/dense/kernel*
T0*
_output_shapes

:@
�
#Adam/update_pi/dense/bias/ApplyAdam	ApplyAdampi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon	Reshape_7*
_output_shapes
:@*
T0*
use_nesterov( *
use_locking( * 
_class
loc:@pi/dense/bias
�
'Adam/update_pi/dense_1/kernel/ApplyAdam	ApplyAdampi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon	Reshape_8*$
_class
loc:@pi/dense_1/kernel*
use_nesterov( *
T0*
use_locking( *
_output_shapes

:@@
�
%Adam/update_pi/dense_1/bias/ApplyAdam	ApplyAdampi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon	Reshape_9*"
_class
loc:@pi/dense_1/bias*
T0*
use_nesterov( *
_output_shapes
:@*
use_locking( 
�
'Adam/update_pi/dense_2/kernel/ApplyAdam	ApplyAdampi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon
Reshape_10*
_output_shapes

:@*
T0*$
_class
loc:@pi/dense_2/kernel*
use_locking( *
use_nesterov( 
�
%Adam/update_pi/dense_2/bias/ApplyAdam	ApplyAdampi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon
Reshape_11*"
_class
loc:@pi/dense_2/bias*
use_nesterov( *
use_locking( *
T0*
_output_shapes
:
�
Adam/mulMulbeta1_power/read
Adam/beta1$^Adam/update_pi/dense/bias/ApplyAdam&^Adam/update_pi/dense/kernel/ApplyAdam&^Adam/update_pi/dense_1/bias/ApplyAdam(^Adam/update_pi/dense_1/kernel/ApplyAdam&^Adam/update_pi/dense_2/bias/ApplyAdam(^Adam/update_pi/dense_2/kernel/ApplyAdam* 
_class
loc:@pi/dense/bias*
T0*
_output_shapes
: 
�
Adam/AssignAssignbeta1_powerAdam/mul*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
use_locking( *
_output_shapes
: 
�

Adam/mul_1Mulbeta2_power/read
Adam/beta2$^Adam/update_pi/dense/bias/ApplyAdam&^Adam/update_pi/dense/kernel/ApplyAdam&^Adam/update_pi/dense_1/bias/ApplyAdam(^Adam/update_pi/dense_1/kernel/ApplyAdam&^Adam/update_pi/dense_2/bias/ApplyAdam(^Adam/update_pi/dense_2/kernel/ApplyAdam* 
_class
loc:@pi/dense/bias*
_output_shapes
: *
T0
�
Adam/Assign_1Assignbeta2_power
Adam/mul_1* 
_class
loc:@pi/dense/bias*
_output_shapes
: *
validate_shape(*
T0*
use_locking( 
�
AdamNoOp^Adam/Assign^Adam/Assign_1$^Adam/update_pi/dense/bias/ApplyAdam&^Adam/update_pi/dense/kernel/ApplyAdam&^Adam/update_pi/dense_1/bias/ApplyAdam(^Adam/update_pi/dense_1/kernel/ApplyAdam&^Adam/update_pi/dense_2/bias/ApplyAdam(^Adam/update_pi/dense_2/kernel/ApplyAdam
j
Reshape_12/shapeConst^Adam*
dtype0*
_output_shapes
:*
valueB:
���������
q

Reshape_12Reshapepi/dense/kernel/readReshape_12/shape*
T0*
_output_shapes	
:�*
Tshape0
j
Reshape_13/shapeConst^Adam*
_output_shapes
:*
dtype0*
valueB:
���������
n

Reshape_13Reshapepi/dense/bias/readReshape_13/shape*
_output_shapes
:@*
Tshape0*
T0
j
Reshape_14/shapeConst^Adam*
_output_shapes
:*
dtype0*
valueB:
���������
s

Reshape_14Reshapepi/dense_1/kernel/readReshape_14/shape*
T0*
Tshape0*
_output_shapes	
:� 
j
Reshape_15/shapeConst^Adam*
valueB:
���������*
_output_shapes
:*
dtype0
p

Reshape_15Reshapepi/dense_1/bias/readReshape_15/shape*
Tshape0*
_output_shapes
:@*
T0
j
Reshape_16/shapeConst^Adam*
valueB:
���������*
_output_shapes
:*
dtype0
s

Reshape_16Reshapepi/dense_2/kernel/readReshape_16/shape*
T0*
_output_shapes	
:�*
Tshape0
j
Reshape_17/shapeConst^Adam*
_output_shapes
:*
valueB:
���������*
dtype0
p

Reshape_17Reshapepi/dense_2/bias/readReshape_17/shape*
Tshape0*
_output_shapes
:*
T0
V
concat_1/axisConst^Adam*
value	B : *
_output_shapes
: *
dtype0
�
concat_1ConcatV2
Reshape_12
Reshape_13
Reshape_14
Reshape_15
Reshape_16
Reshape_17concat_1/axis*

Tidx0*
N*
_output_shapes	
:�%*
T0
h
PyFunc_1PyFuncconcat_1*
_output_shapes
:*
Tin
2*
Tout
2*
token
pyfunc_1
o
Const_6Const^Adam*-
value$B""�  @      @   �      *
_output_shapes
:*
dtype0
Z
split_1/split_dimConst^Adam*
dtype0*
_output_shapes
: *
value	B : 
�
split_1SplitVPyFunc_1Const_6split_1/split_dim*
	num_split*,
_output_shapes
::::::*

Tlen0*
T0
h
Reshape_18/shapeConst^Adam*
dtype0*
_output_shapes
:*
valueB"   @   
g

Reshape_18Reshapesplit_1Reshape_18/shape*
_output_shapes

:@*
T0*
Tshape0
a
Reshape_19/shapeConst^Adam*
dtype0*
_output_shapes
:*
valueB:@
e

Reshape_19Reshape	split_1:1Reshape_19/shape*
_output_shapes
:@*
T0*
Tshape0
h
Reshape_20/shapeConst^Adam*
_output_shapes
:*
dtype0*
valueB"@   @   
i

Reshape_20Reshape	split_1:2Reshape_20/shape*
T0*
Tshape0*
_output_shapes

:@@
a
Reshape_21/shapeConst^Adam*
dtype0*
valueB:@*
_output_shapes
:
e

Reshape_21Reshape	split_1:3Reshape_21/shape*
Tshape0*
_output_shapes
:@*
T0
h
Reshape_22/shapeConst^Adam*
_output_shapes
:*
dtype0*
valueB"@      
i

Reshape_22Reshape	split_1:4Reshape_22/shape*
T0*
_output_shapes

:@*
Tshape0
a
Reshape_23/shapeConst^Adam*
_output_shapes
:*
valueB:*
dtype0
e

Reshape_23Reshape	split_1:5Reshape_23/shape*
T0*
Tshape0*
_output_shapes
:
�
AssignAssignpi/dense/kernel
Reshape_18*"
_class
loc:@pi/dense/kernel*
validate_shape(*
T0*
_output_shapes

:@*
use_locking(
�
Assign_1Assignpi/dense/bias
Reshape_19*
_output_shapes
:@*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(
�
Assign_2Assignpi/dense_1/kernel
Reshape_20*
use_locking(*
validate_shape(*
T0*
_output_shapes

:@@*$
_class
loc:@pi/dense_1/kernel
�
Assign_3Assignpi/dense_1/bias
Reshape_21*
validate_shape(*
use_locking(*
T0*"
_class
loc:@pi/dense_1/bias*
_output_shapes
:@
�
Assign_4Assignpi/dense_2/kernel
Reshape_22*
validate_shape(*
_output_shapes

:@*$
_class
loc:@pi/dense_2/kernel*
T0*
use_locking(
�
Assign_5Assignpi/dense_2/bias
Reshape_23*"
_class
loc:@pi/dense_2/bias*
use_locking(*
_output_shapes
:*
T0*
validate_shape(
Y

group_depsNoOp^Adam^Assign	^Assign_1	^Assign_2	^Assign_3	^Assign_4	^Assign_5
(
group_deps_1NoOp^Adam^group_deps
T
gradients_1/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
Z
gradients_1/grad_ys_0Const*
_output_shapes
: *
dtype0*
valueB
 *  �?
u
gradients_1/FillFillgradients_1/Shapegradients_1/grad_ys_0*
T0*
_output_shapes
: *

index_type0
o
%gradients_1/Mean_1_grad/Reshape/shapeConst*
_output_shapes
:*
valueB:*
dtype0
�
gradients_1/Mean_1_grad/ReshapeReshapegradients_1/Fill%gradients_1/Mean_1_grad/Reshape/shape*
_output_shapes
:*
Tshape0*
T0
`
gradients_1/Mean_1_grad/ShapeShapepow*
T0*
out_type0*
_output_shapes
:
�
gradients_1/Mean_1_grad/TileTilegradients_1/Mean_1_grad/Reshapegradients_1/Mean_1_grad/Shape*
T0*#
_output_shapes
:���������*

Tmultiples0
b
gradients_1/Mean_1_grad/Shape_1Shapepow*
out_type0*
T0*
_output_shapes
:
b
gradients_1/Mean_1_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
g
gradients_1/Mean_1_grad/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
�
gradients_1/Mean_1_grad/ProdProdgradients_1/Mean_1_grad/Shape_1gradients_1/Mean_1_grad/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
i
gradients_1/Mean_1_grad/Const_1Const*
dtype0*
_output_shapes
:*
valueB: 
�
gradients_1/Mean_1_grad/Prod_1Prodgradients_1/Mean_1_grad/Shape_2gradients_1/Mean_1_grad/Const_1*
	keep_dims( *

Tidx0*
_output_shapes
: *
T0
c
!gradients_1/Mean_1_grad/Maximum/yConst*
value	B :*
_output_shapes
: *
dtype0
�
gradients_1/Mean_1_grad/MaximumMaximumgradients_1/Mean_1_grad/Prod_1!gradients_1/Mean_1_grad/Maximum/y*
T0*
_output_shapes
: 
�
 gradients_1/Mean_1_grad/floordivFloorDivgradients_1/Mean_1_grad/Prodgradients_1/Mean_1_grad/Maximum*
_output_shapes
: *
T0
�
gradients_1/Mean_1_grad/CastCast gradients_1/Mean_1_grad/floordiv*

DstT0*
_output_shapes
: *
Truncate( *

SrcT0
�
gradients_1/Mean_1_grad/truedivRealDivgradients_1/Mean_1_grad/Tilegradients_1/Mean_1_grad/Cast*
T0*#
_output_shapes
:���������
_
gradients_1/pow_grad/ShapeShapesub_1*
out_type0*
T0*
_output_shapes
:
_
gradients_1/pow_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
�
*gradients_1/pow_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/pow_grad/Shapegradients_1/pow_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
u
gradients_1/pow_grad/mulMulgradients_1/Mean_1_grad/truedivpow/y*#
_output_shapes
:���������*
T0
_
gradients_1/pow_grad/sub/yConst*
valueB
 *  �?*
_output_shapes
: *
dtype0
c
gradients_1/pow_grad/subSubpow/ygradients_1/pow_grad/sub/y*
_output_shapes
: *
T0
n
gradients_1/pow_grad/PowPowsub_1gradients_1/pow_grad/sub*
T0*#
_output_shapes
:���������
�
gradients_1/pow_grad/mul_1Mulgradients_1/pow_grad/mulgradients_1/pow_grad/Pow*
T0*#
_output_shapes
:���������
�
gradients_1/pow_grad/SumSumgradients_1/pow_grad/mul_1*gradients_1/pow_grad/BroadcastGradientArgs*
	keep_dims( *
T0*
_output_shapes
:*

Tidx0
�
gradients_1/pow_grad/ReshapeReshapegradients_1/pow_grad/Sumgradients_1/pow_grad/Shape*#
_output_shapes
:���������*
T0*
Tshape0
c
gradients_1/pow_grad/Greater/yConst*
dtype0*
valueB
 *    *
_output_shapes
: 
|
gradients_1/pow_grad/GreaterGreatersub_1gradients_1/pow_grad/Greater/y*#
_output_shapes
:���������*
T0
i
$gradients_1/pow_grad/ones_like/ShapeShapesub_1*
out_type0*
T0*
_output_shapes
:
i
$gradients_1/pow_grad/ones_like/ConstConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
gradients_1/pow_grad/ones_likeFill$gradients_1/pow_grad/ones_like/Shape$gradients_1/pow_grad/ones_like/Const*
T0*#
_output_shapes
:���������*

index_type0
�
gradients_1/pow_grad/SelectSelectgradients_1/pow_grad/Greatersub_1gradients_1/pow_grad/ones_like*#
_output_shapes
:���������*
T0
j
gradients_1/pow_grad/LogLoggradients_1/pow_grad/Select*
T0*#
_output_shapes
:���������
a
gradients_1/pow_grad/zeros_like	ZerosLikesub_1*#
_output_shapes
:���������*
T0
�
gradients_1/pow_grad/Select_1Selectgradients_1/pow_grad/Greatergradients_1/pow_grad/Loggradients_1/pow_grad/zeros_like*#
_output_shapes
:���������*
T0
u
gradients_1/pow_grad/mul_2Mulgradients_1/Mean_1_grad/truedivpow*
T0*#
_output_shapes
:���������
�
gradients_1/pow_grad/mul_3Mulgradients_1/pow_grad/mul_2gradients_1/pow_grad/Select_1*
T0*#
_output_shapes
:���������
�
gradients_1/pow_grad/Sum_1Sumgradients_1/pow_grad/mul_3,gradients_1/pow_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
gradients_1/pow_grad/Reshape_1Reshapegradients_1/pow_grad/Sum_1gradients_1/pow_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
m
%gradients_1/pow_grad/tuple/group_depsNoOp^gradients_1/pow_grad/Reshape^gradients_1/pow_grad/Reshape_1
�
-gradients_1/pow_grad/tuple/control_dependencyIdentitygradients_1/pow_grad/Reshape&^gradients_1/pow_grad/tuple/group_deps*#
_output_shapes
:���������*/
_class%
#!loc:@gradients_1/pow_grad/Reshape*
T0
�
/gradients_1/pow_grad/tuple/control_dependency_1Identitygradients_1/pow_grad/Reshape_1&^gradients_1/pow_grad/tuple/group_deps*1
_class'
%#loc:@gradients_1/pow_grad/Reshape_1*
T0*
_output_shapes
: 
i
gradients_1/sub_1_grad/ShapeShapePlaceholder_3*
_output_shapes
:*
T0*
out_type0
g
gradients_1/sub_1_grad/Shape_1Shape	v/Squeeze*
_output_shapes
:*
T0*
out_type0
�
,gradients_1/sub_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/sub_1_grad/Shapegradients_1/sub_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients_1/sub_1_grad/SumSum-gradients_1/pow_grad/tuple/control_dependency,gradients_1/sub_1_grad/BroadcastGradientArgs*
T0*
	keep_dims( *

Tidx0*
_output_shapes
:
�
gradients_1/sub_1_grad/ReshapeReshapegradients_1/sub_1_grad/Sumgradients_1/sub_1_grad/Shape*
T0*
Tshape0*#
_output_shapes
:���������
�
gradients_1/sub_1_grad/Sum_1Sum-gradients_1/pow_grad/tuple/control_dependency.gradients_1/sub_1_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
b
gradients_1/sub_1_grad/NegNeggradients_1/sub_1_grad/Sum_1*
_output_shapes
:*
T0
�
 gradients_1/sub_1_grad/Reshape_1Reshapegradients_1/sub_1_grad/Neggradients_1/sub_1_grad/Shape_1*#
_output_shapes
:���������*
Tshape0*
T0
s
'gradients_1/sub_1_grad/tuple/group_depsNoOp^gradients_1/sub_1_grad/Reshape!^gradients_1/sub_1_grad/Reshape_1
�
/gradients_1/sub_1_grad/tuple/control_dependencyIdentitygradients_1/sub_1_grad/Reshape(^gradients_1/sub_1_grad/tuple/group_deps*#
_output_shapes
:���������*1
_class'
%#loc:@gradients_1/sub_1_grad/Reshape*
T0
�
1gradients_1/sub_1_grad/tuple/control_dependency_1Identity gradients_1/sub_1_grad/Reshape_1(^gradients_1/sub_1_grad/tuple/group_deps*
T0*#
_output_shapes
:���������*3
_class)
'%loc:@gradients_1/sub_1_grad/Reshape_1
q
 gradients_1/v/Squeeze_grad/ShapeShapev/dense_2/BiasAdd*
out_type0*
_output_shapes
:*
T0
�
"gradients_1/v/Squeeze_grad/ReshapeReshape1gradients_1/sub_1_grad/tuple/control_dependency_1 gradients_1/v/Squeeze_grad/Shape*
Tshape0*'
_output_shapes
:���������*
T0
�
.gradients_1/v/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad"gradients_1/v/Squeeze_grad/Reshape*
_output_shapes
:*
data_formatNHWC*
T0
�
3gradients_1/v/dense_2/BiasAdd_grad/tuple/group_depsNoOp#^gradients_1/v/Squeeze_grad/Reshape/^gradients_1/v/dense_2/BiasAdd_grad/BiasAddGrad
�
;gradients_1/v/dense_2/BiasAdd_grad/tuple/control_dependencyIdentity"gradients_1/v/Squeeze_grad/Reshape4^gradients_1/v/dense_2/BiasAdd_grad/tuple/group_deps*
T0*'
_output_shapes
:���������*5
_class+
)'loc:@gradients_1/v/Squeeze_grad/Reshape
�
=gradients_1/v/dense_2/BiasAdd_grad/tuple/control_dependency_1Identity.gradients_1/v/dense_2/BiasAdd_grad/BiasAddGrad4^gradients_1/v/dense_2/BiasAdd_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients_1/v/dense_2/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
�
(gradients_1/v/dense_2/MatMul_grad/MatMulMatMul;gradients_1/v/dense_2/BiasAdd_grad/tuple/control_dependencyv/dense_2/kernel/read*
transpose_a( *
T0*
transpose_b(*'
_output_shapes
:���������@
�
*gradients_1/v/dense_2/MatMul_grad/MatMul_1MatMulv/dense_1/Tanh;gradients_1/v/dense_2/BiasAdd_grad/tuple/control_dependency*
T0*
transpose_a(*
transpose_b( *
_output_shapes

:@
�
2gradients_1/v/dense_2/MatMul_grad/tuple/group_depsNoOp)^gradients_1/v/dense_2/MatMul_grad/MatMul+^gradients_1/v/dense_2/MatMul_grad/MatMul_1
�
:gradients_1/v/dense_2/MatMul_grad/tuple/control_dependencyIdentity(gradients_1/v/dense_2/MatMul_grad/MatMul3^gradients_1/v/dense_2/MatMul_grad/tuple/group_deps*;
_class1
/-loc:@gradients_1/v/dense_2/MatMul_grad/MatMul*'
_output_shapes
:���������@*
T0
�
<gradients_1/v/dense_2/MatMul_grad/tuple/control_dependency_1Identity*gradients_1/v/dense_2/MatMul_grad/MatMul_13^gradients_1/v/dense_2/MatMul_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients_1/v/dense_2/MatMul_grad/MatMul_1*
_output_shapes

:@
�
(gradients_1/v/dense_1/Tanh_grad/TanhGradTanhGradv/dense_1/Tanh:gradients_1/v/dense_2/MatMul_grad/tuple/control_dependency*
T0*'
_output_shapes
:���������@
�
.gradients_1/v/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad(gradients_1/v/dense_1/Tanh_grad/TanhGrad*
T0*
data_formatNHWC*
_output_shapes
:@
�
3gradients_1/v/dense_1/BiasAdd_grad/tuple/group_depsNoOp/^gradients_1/v/dense_1/BiasAdd_grad/BiasAddGrad)^gradients_1/v/dense_1/Tanh_grad/TanhGrad
�
;gradients_1/v/dense_1/BiasAdd_grad/tuple/control_dependencyIdentity(gradients_1/v/dense_1/Tanh_grad/TanhGrad4^gradients_1/v/dense_1/BiasAdd_grad/tuple/group_deps*'
_output_shapes
:���������@*
T0*;
_class1
/-loc:@gradients_1/v/dense_1/Tanh_grad/TanhGrad
�
=gradients_1/v/dense_1/BiasAdd_grad/tuple/control_dependency_1Identity.gradients_1/v/dense_1/BiasAdd_grad/BiasAddGrad4^gradients_1/v/dense_1/BiasAdd_grad/tuple/group_deps*
_output_shapes
:@*
T0*A
_class7
53loc:@gradients_1/v/dense_1/BiasAdd_grad/BiasAddGrad
�
(gradients_1/v/dense_1/MatMul_grad/MatMulMatMul;gradients_1/v/dense_1/BiasAdd_grad/tuple/control_dependencyv/dense_1/kernel/read*
transpose_a( *
T0*
transpose_b(*'
_output_shapes
:���������@
�
*gradients_1/v/dense_1/MatMul_grad/MatMul_1MatMulv/dense/Tanh;gradients_1/v/dense_1/BiasAdd_grad/tuple/control_dependency*
transpose_a(*
transpose_b( *
T0*
_output_shapes

:@@
�
2gradients_1/v/dense_1/MatMul_grad/tuple/group_depsNoOp)^gradients_1/v/dense_1/MatMul_grad/MatMul+^gradients_1/v/dense_1/MatMul_grad/MatMul_1
�
:gradients_1/v/dense_1/MatMul_grad/tuple/control_dependencyIdentity(gradients_1/v/dense_1/MatMul_grad/MatMul3^gradients_1/v/dense_1/MatMul_grad/tuple/group_deps*;
_class1
/-loc:@gradients_1/v/dense_1/MatMul_grad/MatMul*'
_output_shapes
:���������@*
T0
�
<gradients_1/v/dense_1/MatMul_grad/tuple/control_dependency_1Identity*gradients_1/v/dense_1/MatMul_grad/MatMul_13^gradients_1/v/dense_1/MatMul_grad/tuple/group_deps*
_output_shapes

:@@*=
_class3
1/loc:@gradients_1/v/dense_1/MatMul_grad/MatMul_1*
T0
�
&gradients_1/v/dense/Tanh_grad/TanhGradTanhGradv/dense/Tanh:gradients_1/v/dense_1/MatMul_grad/tuple/control_dependency*
T0*'
_output_shapes
:���������@
�
,gradients_1/v/dense/BiasAdd_grad/BiasAddGradBiasAddGrad&gradients_1/v/dense/Tanh_grad/TanhGrad*
data_formatNHWC*
_output_shapes
:@*
T0
�
1gradients_1/v/dense/BiasAdd_grad/tuple/group_depsNoOp-^gradients_1/v/dense/BiasAdd_grad/BiasAddGrad'^gradients_1/v/dense/Tanh_grad/TanhGrad
�
9gradients_1/v/dense/BiasAdd_grad/tuple/control_dependencyIdentity&gradients_1/v/dense/Tanh_grad/TanhGrad2^gradients_1/v/dense/BiasAdd_grad/tuple/group_deps*9
_class/
-+loc:@gradients_1/v/dense/Tanh_grad/TanhGrad*
T0*'
_output_shapes
:���������@
�
;gradients_1/v/dense/BiasAdd_grad/tuple/control_dependency_1Identity,gradients_1/v/dense/BiasAdd_grad/BiasAddGrad2^gradients_1/v/dense/BiasAdd_grad/tuple/group_deps*
_output_shapes
:@*?
_class5
31loc:@gradients_1/v/dense/BiasAdd_grad/BiasAddGrad*
T0
�
&gradients_1/v/dense/MatMul_grad/MatMulMatMul9gradients_1/v/dense/BiasAdd_grad/tuple/control_dependencyv/dense/kernel/read*
transpose_b(*
T0*
transpose_a( *'
_output_shapes
:���������
�
(gradients_1/v/dense/MatMul_grad/MatMul_1MatMulPlaceholder9gradients_1/v/dense/BiasAdd_grad/tuple/control_dependency*
transpose_a(*
T0*
transpose_b( *
_output_shapes

:@
�
0gradients_1/v/dense/MatMul_grad/tuple/group_depsNoOp'^gradients_1/v/dense/MatMul_grad/MatMul)^gradients_1/v/dense/MatMul_grad/MatMul_1
�
8gradients_1/v/dense/MatMul_grad/tuple/control_dependencyIdentity&gradients_1/v/dense/MatMul_grad/MatMul1^gradients_1/v/dense/MatMul_grad/tuple/group_deps*
T0*9
_class/
-+loc:@gradients_1/v/dense/MatMul_grad/MatMul*'
_output_shapes
:���������
�
:gradients_1/v/dense/MatMul_grad/tuple/control_dependency_1Identity(gradients_1/v/dense/MatMul_grad/MatMul_11^gradients_1/v/dense/MatMul_grad/tuple/group_deps*
_output_shapes

:@*;
_class1
/-loc:@gradients_1/v/dense/MatMul_grad/MatMul_1*
T0
c
Reshape_24/shapeConst*
_output_shapes
:*
dtype0*
valueB:
���������
�

Reshape_24Reshape:gradients_1/v/dense/MatMul_grad/tuple/control_dependency_1Reshape_24/shape*
T0*
Tshape0*
_output_shapes	
:�
c
Reshape_25/shapeConst*
_output_shapes
:*
valueB:
���������*
dtype0
�

Reshape_25Reshape;gradients_1/v/dense/BiasAdd_grad/tuple/control_dependency_1Reshape_25/shape*
_output_shapes
:@*
T0*
Tshape0
c
Reshape_26/shapeConst*
valueB:
���������*
_output_shapes
:*
dtype0
�

Reshape_26Reshape<gradients_1/v/dense_1/MatMul_grad/tuple/control_dependency_1Reshape_26/shape*
T0*
Tshape0*
_output_shapes	
:� 
c
Reshape_27/shapeConst*
_output_shapes
:*
valueB:
���������*
dtype0
�

Reshape_27Reshape=gradients_1/v/dense_1/BiasAdd_grad/tuple/control_dependency_1Reshape_27/shape*
T0*
_output_shapes
:@*
Tshape0
c
Reshape_28/shapeConst*
dtype0*
valueB:
���������*
_output_shapes
:
�

Reshape_28Reshape<gradients_1/v/dense_2/MatMul_grad/tuple/control_dependency_1Reshape_28/shape*
_output_shapes
:@*
Tshape0*
T0
c
Reshape_29/shapeConst*
valueB:
���������*
dtype0*
_output_shapes
:
�

Reshape_29Reshape=gradients_1/v/dense_2/BiasAdd_grad/tuple/control_dependency_1Reshape_29/shape*
_output_shapes
:*
Tshape0*
T0
O
concat_2/axisConst*
dtype0*
_output_shapes
: *
value	B : 
�
concat_2ConcatV2
Reshape_24
Reshape_25
Reshape_26
Reshape_27
Reshape_28
Reshape_29concat_2/axis*
_output_shapes	
:�$*

Tidx0*
N*
T0
k
PyFunc_2PyFuncconcat_2*
_output_shapes	
:�$*
token
pyfunc_2*
Tout
2*
Tin
2
h
Const_7Const*
dtype0*
_output_shapes
:*-
value$B""�  @      @   @      
S
split_2/split_dimConst*
dtype0*
_output_shapes
: *
value	B : 
�
split_2SplitVPyFunc_2Const_7split_2/split_dim*:
_output_shapes(
&:�:@:� :@:@:*
T0*

Tlen0*
	num_split
a
Reshape_30/shapeConst*
_output_shapes
:*
valueB"   @   *
dtype0
g

Reshape_30Reshapesplit_2Reshape_30/shape*
T0*
Tshape0*
_output_shapes

:@
Z
Reshape_31/shapeConst*
_output_shapes
:*
dtype0*
valueB:@
e

Reshape_31Reshape	split_2:1Reshape_31/shape*
_output_shapes
:@*
T0*
Tshape0
a
Reshape_32/shapeConst*
dtype0*
_output_shapes
:*
valueB"@   @   
i

Reshape_32Reshape	split_2:2Reshape_32/shape*
T0*
Tshape0*
_output_shapes

:@@
Z
Reshape_33/shapeConst*
valueB:@*
dtype0*
_output_shapes
:
e

Reshape_33Reshape	split_2:3Reshape_33/shape*
Tshape0*
_output_shapes
:@*
T0
a
Reshape_34/shapeConst*
_output_shapes
:*
valueB"@      *
dtype0
i

Reshape_34Reshape	split_2:4Reshape_34/shape*
T0*
_output_shapes

:@*
Tshape0
Z
Reshape_35/shapeConst*
_output_shapes
:*
dtype0*
valueB:
e

Reshape_35Reshape	split_2:5Reshape_35/shape*
Tshape0*
_output_shapes
:*
T0
�
beta1_power_1/initial_valueConst*
dtype0*
_class
loc:@v/dense/bias*
_output_shapes
: *
valueB
 *fff?
�
beta1_power_1
VariableV2*
_output_shapes
: *
shape: *
_class
loc:@v/dense/bias*
	container *
dtype0*
shared_name 
�
beta1_power_1/AssignAssignbeta1_power_1beta1_power_1/initial_value*
_class
loc:@v/dense/bias*
T0*
use_locking(*
validate_shape(*
_output_shapes
: 
o
beta1_power_1/readIdentitybeta1_power_1*
_class
loc:@v/dense/bias*
T0*
_output_shapes
: 
�
beta2_power_1/initial_valueConst*
dtype0*
_class
loc:@v/dense/bias*
_output_shapes
: *
valueB
 *w�?
�
beta2_power_1
VariableV2*
_class
loc:@v/dense/bias*
shared_name *
shape: *
dtype0*
	container *
_output_shapes
: 
�
beta2_power_1/AssignAssignbeta2_power_1beta2_power_1/initial_value*
_output_shapes
: *
_class
loc:@v/dense/bias*
use_locking(*
validate_shape(*
T0
o
beta2_power_1/readIdentitybeta2_power_1*
_output_shapes
: *
T0*
_class
loc:@v/dense/bias
�
%v/dense/kernel/Adam/Initializer/zerosConst*
dtype0*
_output_shapes

:@*
valueB@*    *!
_class
loc:@v/dense/kernel
�
v/dense/kernel/Adam
VariableV2*!
_class
loc:@v/dense/kernel*
dtype0*
	container *
_output_shapes

:@*
shape
:@*
shared_name 
�
v/dense/kernel/Adam/AssignAssignv/dense/kernel/Adam%v/dense/kernel/Adam/Initializer/zeros*
_output_shapes

:@*!
_class
loc:@v/dense/kernel*
validate_shape(*
use_locking(*
T0
�
v/dense/kernel/Adam/readIdentityv/dense/kernel/Adam*
T0*!
_class
loc:@v/dense/kernel*
_output_shapes

:@
�
'v/dense/kernel/Adam_1/Initializer/zerosConst*
_output_shapes

:@*!
_class
loc:@v/dense/kernel*
valueB@*    *
dtype0
�
v/dense/kernel/Adam_1
VariableV2*
shape
:@*
	container *
_output_shapes

:@*
shared_name *!
_class
loc:@v/dense/kernel*
dtype0
�
v/dense/kernel/Adam_1/AssignAssignv/dense/kernel/Adam_1'v/dense/kernel/Adam_1/Initializer/zeros*
use_locking(*
validate_shape(*!
_class
loc:@v/dense/kernel*
_output_shapes

:@*
T0
�
v/dense/kernel/Adam_1/readIdentityv/dense/kernel/Adam_1*
T0*
_output_shapes

:@*!
_class
loc:@v/dense/kernel
�
#v/dense/bias/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
:@*
_class
loc:@v/dense/bias*
valueB@*    
�
v/dense/bias/Adam
VariableV2*
	container *
_output_shapes
:@*
dtype0*
shared_name *
shape:@*
_class
loc:@v/dense/bias
�
v/dense/bias/Adam/AssignAssignv/dense/bias/Adam#v/dense/bias/Adam/Initializer/zeros*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
:@*
T0*
use_locking(
{
v/dense/bias/Adam/readIdentityv/dense/bias/Adam*
T0*
_output_shapes
:@*
_class
loc:@v/dense/bias
�
%v/dense/bias/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes
:@*
_class
loc:@v/dense/bias*
valueB@*    
�
v/dense/bias/Adam_1
VariableV2*
_class
loc:@v/dense/bias*
_output_shapes
:@*
dtype0*
shape:@*
shared_name *
	container 
�
v/dense/bias/Adam_1/AssignAssignv/dense/bias/Adam_1%v/dense/bias/Adam_1/Initializer/zeros*
T0*
use_locking(*
_output_shapes
:@*
validate_shape(*
_class
loc:@v/dense/bias

v/dense/bias/Adam_1/readIdentityv/dense/bias/Adam_1*
_output_shapes
:@*
T0*
_class
loc:@v/dense/bias
�
7v/dense_1/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"@   @   *#
_class
loc:@v/dense_1/kernel
�
-v/dense_1/kernel/Adam/Initializer/zeros/ConstConst*#
_class
loc:@v/dense_1/kernel*
valueB
 *    *
_output_shapes
: *
dtype0
�
'v/dense_1/kernel/Adam/Initializer/zerosFill7v/dense_1/kernel/Adam/Initializer/zeros/shape_as_tensor-v/dense_1/kernel/Adam/Initializer/zeros/Const*
_output_shapes

:@@*
T0*#
_class
loc:@v/dense_1/kernel*

index_type0
�
v/dense_1/kernel/Adam
VariableV2*
shape
:@@*
dtype0*
	container *#
_class
loc:@v/dense_1/kernel*
shared_name *
_output_shapes

:@@
�
v/dense_1/kernel/Adam/AssignAssignv/dense_1/kernel/Adam'v/dense_1/kernel/Adam/Initializer/zeros*
validate_shape(*
_output_shapes

:@@*
T0*
use_locking(*#
_class
loc:@v/dense_1/kernel
�
v/dense_1/kernel/Adam/readIdentityv/dense_1/kernel/Adam*
_output_shapes

:@@*#
_class
loc:@v/dense_1/kernel*
T0
�
9v/dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*#
_class
loc:@v/dense_1/kernel*
dtype0*
_output_shapes
:*
valueB"@   @   
�
/v/dense_1/kernel/Adam_1/Initializer/zeros/ConstConst*
_output_shapes
: *#
_class
loc:@v/dense_1/kernel*
dtype0*
valueB
 *    
�
)v/dense_1/kernel/Adam_1/Initializer/zerosFill9v/dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensor/v/dense_1/kernel/Adam_1/Initializer/zeros/Const*#
_class
loc:@v/dense_1/kernel*
T0*

index_type0*
_output_shapes

:@@
�
v/dense_1/kernel/Adam_1
VariableV2*
	container *#
_class
loc:@v/dense_1/kernel*
shared_name *
shape
:@@*
_output_shapes

:@@*
dtype0
�
v/dense_1/kernel/Adam_1/AssignAssignv/dense_1/kernel/Adam_1)v/dense_1/kernel/Adam_1/Initializer/zeros*#
_class
loc:@v/dense_1/kernel*
validate_shape(*
_output_shapes

:@@*
T0*
use_locking(
�
v/dense_1/kernel/Adam_1/readIdentityv/dense_1/kernel/Adam_1*
T0*#
_class
loc:@v/dense_1/kernel*
_output_shapes

:@@
�
%v/dense_1/bias/Adam/Initializer/zerosConst*
_output_shapes
:@*
dtype0*!
_class
loc:@v/dense_1/bias*
valueB@*    
�
v/dense_1/bias/Adam
VariableV2*
_output_shapes
:@*
shape:@*
shared_name *!
_class
loc:@v/dense_1/bias*
	container *
dtype0
�
v/dense_1/bias/Adam/AssignAssignv/dense_1/bias/Adam%v/dense_1/bias/Adam/Initializer/zeros*
T0*
_output_shapes
:@*
validate_shape(*!
_class
loc:@v/dense_1/bias*
use_locking(
�
v/dense_1/bias/Adam/readIdentityv/dense_1/bias/Adam*
T0*!
_class
loc:@v/dense_1/bias*
_output_shapes
:@
�
'v/dense_1/bias/Adam_1/Initializer/zerosConst*
valueB@*    *!
_class
loc:@v/dense_1/bias*
_output_shapes
:@*
dtype0
�
v/dense_1/bias/Adam_1
VariableV2*
shape:@*
_output_shapes
:@*
shared_name *
	container *
dtype0*!
_class
loc:@v/dense_1/bias
�
v/dense_1/bias/Adam_1/AssignAssignv/dense_1/bias/Adam_1'v/dense_1/bias/Adam_1/Initializer/zeros*!
_class
loc:@v/dense_1/bias*
T0*
_output_shapes
:@*
validate_shape(*
use_locking(
�
v/dense_1/bias/Adam_1/readIdentityv/dense_1/bias/Adam_1*
_output_shapes
:@*
T0*!
_class
loc:@v/dense_1/bias
�
'v/dense_2/kernel/Adam/Initializer/zerosConst*
valueB@*    *
_output_shapes

:@*
dtype0*#
_class
loc:@v/dense_2/kernel
�
v/dense_2/kernel/Adam
VariableV2*#
_class
loc:@v/dense_2/kernel*
_output_shapes

:@*
dtype0*
	container *
shape
:@*
shared_name 
�
v/dense_2/kernel/Adam/AssignAssignv/dense_2/kernel/Adam'v/dense_2/kernel/Adam/Initializer/zeros*
T0*#
_class
loc:@v/dense_2/kernel*
validate_shape(*
use_locking(*
_output_shapes

:@
�
v/dense_2/kernel/Adam/readIdentityv/dense_2/kernel/Adam*#
_class
loc:@v/dense_2/kernel*
_output_shapes

:@*
T0
�
)v/dense_2/kernel/Adam_1/Initializer/zerosConst*
dtype0*#
_class
loc:@v/dense_2/kernel*
_output_shapes

:@*
valueB@*    
�
v/dense_2/kernel/Adam_1
VariableV2*
_output_shapes

:@*
dtype0*#
_class
loc:@v/dense_2/kernel*
	container *
shared_name *
shape
:@
�
v/dense_2/kernel/Adam_1/AssignAssignv/dense_2/kernel/Adam_1)v/dense_2/kernel/Adam_1/Initializer/zeros*
validate_shape(*#
_class
loc:@v/dense_2/kernel*
use_locking(*
T0*
_output_shapes

:@
�
v/dense_2/kernel/Adam_1/readIdentityv/dense_2/kernel/Adam_1*#
_class
loc:@v/dense_2/kernel*
_output_shapes

:@*
T0
�
%v/dense_2/bias/Adam/Initializer/zerosConst*
dtype0*
valueB*    *
_output_shapes
:*!
_class
loc:@v/dense_2/bias
�
v/dense_2/bias/Adam
VariableV2*
	container *
shape:*!
_class
loc:@v/dense_2/bias*
_output_shapes
:*
shared_name *
dtype0
�
v/dense_2/bias/Adam/AssignAssignv/dense_2/bias/Adam%v/dense_2/bias/Adam/Initializer/zeros*
T0*
use_locking(*!
_class
loc:@v/dense_2/bias*
validate_shape(*
_output_shapes
:
�
v/dense_2/bias/Adam/readIdentityv/dense_2/bias/Adam*
_output_shapes
:*
T0*!
_class
loc:@v/dense_2/bias
�
'v/dense_2/bias/Adam_1/Initializer/zerosConst*
valueB*    *
_output_shapes
:*
dtype0*!
_class
loc:@v/dense_2/bias
�
v/dense_2/bias/Adam_1
VariableV2*
shape:*
dtype0*
	container *
_output_shapes
:*
shared_name *!
_class
loc:@v/dense_2/bias
�
v/dense_2/bias/Adam_1/AssignAssignv/dense_2/bias/Adam_1'v/dense_2/bias/Adam_1/Initializer/zeros*
use_locking(*
_output_shapes
:*
T0*!
_class
loc:@v/dense_2/bias*
validate_shape(
�
v/dense_2/bias/Adam_1/readIdentityv/dense_2/bias/Adam_1*
_output_shapes
:*!
_class
loc:@v/dense_2/bias*
T0
Y
Adam_1/learning_rateConst*
_output_shapes
: *
valueB
 *o�:*
dtype0
Q
Adam_1/beta1Const*
valueB
 *fff?*
_output_shapes
: *
dtype0
Q
Adam_1/beta2Const*
dtype0*
valueB
 *w�?*
_output_shapes
: 
S
Adam_1/epsilonConst*
valueB
 *w�+2*
dtype0*
_output_shapes
: 
�
&Adam_1/update_v/dense/kernel/ApplyAdam	ApplyAdamv/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon
Reshape_30*
_output_shapes

:@*!
_class
loc:@v/dense/kernel*
use_locking( *
use_nesterov( *
T0
�
$Adam_1/update_v/dense/bias/ApplyAdam	ApplyAdamv/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon
Reshape_31*
use_nesterov( *
_class
loc:@v/dense/bias*
T0*
_output_shapes
:@*
use_locking( 
�
(Adam_1/update_v/dense_1/kernel/ApplyAdam	ApplyAdamv/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon
Reshape_32*#
_class
loc:@v/dense_1/kernel*
T0*
use_locking( *
_output_shapes

:@@*
use_nesterov( 
�
&Adam_1/update_v/dense_1/bias/ApplyAdam	ApplyAdamv/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon
Reshape_33*
T0*
use_locking( *
_output_shapes
:@*!
_class
loc:@v/dense_1/bias*
use_nesterov( 
�
(Adam_1/update_v/dense_2/kernel/ApplyAdam	ApplyAdamv/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon
Reshape_34*#
_class
loc:@v/dense_2/kernel*
_output_shapes

:@*
use_nesterov( *
T0*
use_locking( 
�
&Adam_1/update_v/dense_2/bias/ApplyAdam	ApplyAdamv/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon
Reshape_35*
use_nesterov( *
_output_shapes
:*!
_class
loc:@v/dense_2/bias*
use_locking( *
T0
�

Adam_1/mulMulbeta1_power_1/readAdam_1/beta1%^Adam_1/update_v/dense/bias/ApplyAdam'^Adam_1/update_v/dense/kernel/ApplyAdam'^Adam_1/update_v/dense_1/bias/ApplyAdam)^Adam_1/update_v/dense_1/kernel/ApplyAdam'^Adam_1/update_v/dense_2/bias/ApplyAdam)^Adam_1/update_v/dense_2/kernel/ApplyAdam*
_output_shapes
: *
T0*
_class
loc:@v/dense/bias
�
Adam_1/AssignAssignbeta1_power_1
Adam_1/mul*
validate_shape(*
use_locking( *
_class
loc:@v/dense/bias*
T0*
_output_shapes
: 
�
Adam_1/mul_1Mulbeta2_power_1/readAdam_1/beta2%^Adam_1/update_v/dense/bias/ApplyAdam'^Adam_1/update_v/dense/kernel/ApplyAdam'^Adam_1/update_v/dense_1/bias/ApplyAdam)^Adam_1/update_v/dense_1/kernel/ApplyAdam'^Adam_1/update_v/dense_2/bias/ApplyAdam)^Adam_1/update_v/dense_2/kernel/ApplyAdam*
_output_shapes
: *
T0*
_class
loc:@v/dense/bias
�
Adam_1/Assign_1Assignbeta2_power_1Adam_1/mul_1*
_output_shapes
: *
_class
loc:@v/dense/bias*
use_locking( *
T0*
validate_shape(
�
Adam_1NoOp^Adam_1/Assign^Adam_1/Assign_1%^Adam_1/update_v/dense/bias/ApplyAdam'^Adam_1/update_v/dense/kernel/ApplyAdam'^Adam_1/update_v/dense_1/bias/ApplyAdam)^Adam_1/update_v/dense_1/kernel/ApplyAdam'^Adam_1/update_v/dense_2/bias/ApplyAdam)^Adam_1/update_v/dense_2/kernel/ApplyAdam
l
Reshape_36/shapeConst^Adam_1*
dtype0*
valueB:
���������*
_output_shapes
:
p

Reshape_36Reshapev/dense/kernel/readReshape_36/shape*
T0*
_output_shapes	
:�*
Tshape0
l
Reshape_37/shapeConst^Adam_1*
dtype0*
_output_shapes
:*
valueB:
���������
m

Reshape_37Reshapev/dense/bias/readReshape_37/shape*
Tshape0*
T0*
_output_shapes
:@
l
Reshape_38/shapeConst^Adam_1*
_output_shapes
:*
valueB:
���������*
dtype0
r

Reshape_38Reshapev/dense_1/kernel/readReshape_38/shape*
_output_shapes	
:� *
Tshape0*
T0
l
Reshape_39/shapeConst^Adam_1*
valueB:
���������*
dtype0*
_output_shapes
:
o

Reshape_39Reshapev/dense_1/bias/readReshape_39/shape*
T0*
_output_shapes
:@*
Tshape0
l
Reshape_40/shapeConst^Adam_1*
valueB:
���������*
dtype0*
_output_shapes
:
q

Reshape_40Reshapev/dense_2/kernel/readReshape_40/shape*
T0*
_output_shapes
:@*
Tshape0
l
Reshape_41/shapeConst^Adam_1*
_output_shapes
:*
dtype0*
valueB:
���������
o

Reshape_41Reshapev/dense_2/bias/readReshape_41/shape*
Tshape0*
_output_shapes
:*
T0
X
concat_3/axisConst^Adam_1*
dtype0*
value	B : *
_output_shapes
: 
�
concat_3ConcatV2
Reshape_36
Reshape_37
Reshape_38
Reshape_39
Reshape_40
Reshape_41concat_3/axis*

Tidx0*
_output_shapes	
:�$*
N*
T0
h
PyFunc_3PyFuncconcat_3*
token
pyfunc_3*
_output_shapes
:*
Tin
2*
Tout
2
q
Const_8Const^Adam_1*
_output_shapes
:*-
value$B""�  @      @   @      *
dtype0
\
split_3/split_dimConst^Adam_1*
value	B : *
_output_shapes
: *
dtype0
�
split_3SplitVPyFunc_3Const_8split_3/split_dim*

Tlen0*
T0*
	num_split*,
_output_shapes
::::::
j
Reshape_42/shapeConst^Adam_1*
dtype0*
valueB"   @   *
_output_shapes
:
g

Reshape_42Reshapesplit_3Reshape_42/shape*
_output_shapes

:@*
Tshape0*
T0
c
Reshape_43/shapeConst^Adam_1*
_output_shapes
:*
dtype0*
valueB:@
e

Reshape_43Reshape	split_3:1Reshape_43/shape*
Tshape0*
_output_shapes
:@*
T0
j
Reshape_44/shapeConst^Adam_1*
_output_shapes
:*
valueB"@   @   *
dtype0
i

Reshape_44Reshape	split_3:2Reshape_44/shape*
_output_shapes

:@@*
T0*
Tshape0
c
Reshape_45/shapeConst^Adam_1*
valueB:@*
dtype0*
_output_shapes
:
e

Reshape_45Reshape	split_3:3Reshape_45/shape*
T0*
_output_shapes
:@*
Tshape0
j
Reshape_46/shapeConst^Adam_1*
valueB"@      *
_output_shapes
:*
dtype0
i

Reshape_46Reshape	split_3:4Reshape_46/shape*
Tshape0*
_output_shapes

:@*
T0
c
Reshape_47/shapeConst^Adam_1*
dtype0*
_output_shapes
:*
valueB:
e

Reshape_47Reshape	split_3:5Reshape_47/shape*
_output_shapes
:*
T0*
Tshape0
�
Assign_6Assignv/dense/kernel
Reshape_42*
_output_shapes

:@*
use_locking(*
T0*!
_class
loc:@v/dense/kernel*
validate_shape(
�
Assign_7Assignv/dense/bias
Reshape_43*
validate_shape(*
use_locking(*
_output_shapes
:@*
T0*
_class
loc:@v/dense/bias
�
Assign_8Assignv/dense_1/kernel
Reshape_44*
_output_shapes

:@@*
validate_shape(*
T0*#
_class
loc:@v/dense_1/kernel*
use_locking(
�
Assign_9Assignv/dense_1/bias
Reshape_45*
use_locking(*
T0*
_output_shapes
:@*
validate_shape(*!
_class
loc:@v/dense_1/bias
�
	Assign_10Assignv/dense_2/kernel
Reshape_46*
_output_shapes

:@*
use_locking(*
T0*#
_class
loc:@v/dense_2/kernel*
validate_shape(
�
	Assign_11Assignv/dense_2/bias
Reshape_47*
use_locking(*
T0*!
_class
loc:@v/dense_2/bias*
validate_shape(*
_output_shapes
:
a
group_deps_2NoOp^Adam_1
^Assign_10
^Assign_11	^Assign_6	^Assign_7	^Assign_8	^Assign_9
,
group_deps_3NoOp^Adam_1^group_deps_2
�
initNoOp^beta1_power/Assign^beta1_power_1/Assign^beta2_power/Assign^beta2_power_1/Assign^pi/dense/bias/Adam/Assign^pi/dense/bias/Adam_1/Assign^pi/dense/bias/Assign^pi/dense/kernel/Adam/Assign^pi/dense/kernel/Adam_1/Assign^pi/dense/kernel/Assign^pi/dense_1/bias/Adam/Assign^pi/dense_1/bias/Adam_1/Assign^pi/dense_1/bias/Assign^pi/dense_1/kernel/Adam/Assign ^pi/dense_1/kernel/Adam_1/Assign^pi/dense_1/kernel/Assign^pi/dense_2/bias/Adam/Assign^pi/dense_2/bias/Adam_1/Assign^pi/dense_2/bias/Assign^pi/dense_2/kernel/Adam/Assign ^pi/dense_2/kernel/Adam_1/Assign^pi/dense_2/kernel/Assign^v/dense/bias/Adam/Assign^v/dense/bias/Adam_1/Assign^v/dense/bias/Assign^v/dense/kernel/Adam/Assign^v/dense/kernel/Adam_1/Assign^v/dense/kernel/Assign^v/dense_1/bias/Adam/Assign^v/dense_1/bias/Adam_1/Assign^v/dense_1/bias/Assign^v/dense_1/kernel/Adam/Assign^v/dense_1/kernel/Adam_1/Assign^v/dense_1/kernel/Assign^v/dense_2/bias/Adam/Assign^v/dense_2/bias/Adam_1/Assign^v/dense_2/bias/Assign^v/dense_2/kernel/Adam/Assign^v/dense_2/kernel/Adam_1/Assign^v/dense_2/kernel/Assign
c
Reshape_48/shapeConst*
dtype0*
valueB:
���������*
_output_shapes
:
q

Reshape_48Reshapepi/dense/kernel/readReshape_48/shape*
_output_shapes	
:�*
T0*
Tshape0
c
Reshape_49/shapeConst*
valueB:
���������*
_output_shapes
:*
dtype0
n

Reshape_49Reshapepi/dense/bias/readReshape_49/shape*
Tshape0*
_output_shapes
:@*
T0
c
Reshape_50/shapeConst*
_output_shapes
:*
dtype0*
valueB:
���������
s

Reshape_50Reshapepi/dense_1/kernel/readReshape_50/shape*
Tshape0*
_output_shapes	
:� *
T0
c
Reshape_51/shapeConst*
dtype0*
_output_shapes
:*
valueB:
���������
p

Reshape_51Reshapepi/dense_1/bias/readReshape_51/shape*
T0*
Tshape0*
_output_shapes
:@
c
Reshape_52/shapeConst*
valueB:
���������*
dtype0*
_output_shapes
:
s

Reshape_52Reshapepi/dense_2/kernel/readReshape_52/shape*
T0*
Tshape0*
_output_shapes	
:�
c
Reshape_53/shapeConst*
_output_shapes
:*
dtype0*
valueB:
���������
p

Reshape_53Reshapepi/dense_2/bias/readReshape_53/shape*
_output_shapes
:*
Tshape0*
T0
c
Reshape_54/shapeConst*
dtype0*
_output_shapes
:*
valueB:
���������
p

Reshape_54Reshapev/dense/kernel/readReshape_54/shape*
T0*
_output_shapes	
:�*
Tshape0
c
Reshape_55/shapeConst*
_output_shapes
:*
dtype0*
valueB:
���������
m

Reshape_55Reshapev/dense/bias/readReshape_55/shape*
Tshape0*
T0*
_output_shapes
:@
c
Reshape_56/shapeConst*
dtype0*
valueB:
���������*
_output_shapes
:
r

Reshape_56Reshapev/dense_1/kernel/readReshape_56/shape*
T0*
_output_shapes	
:� *
Tshape0
c
Reshape_57/shapeConst*
_output_shapes
:*
valueB:
���������*
dtype0
o

Reshape_57Reshapev/dense_1/bias/readReshape_57/shape*
Tshape0*
_output_shapes
:@*
T0
c
Reshape_58/shapeConst*
valueB:
���������*
_output_shapes
:*
dtype0
q

Reshape_58Reshapev/dense_2/kernel/readReshape_58/shape*
_output_shapes
:@*
T0*
Tshape0
c
Reshape_59/shapeConst*
dtype0*
_output_shapes
:*
valueB:
���������
o

Reshape_59Reshapev/dense_2/bias/readReshape_59/shape*
T0*
Tshape0*
_output_shapes
:
c
Reshape_60/shapeConst*
dtype0*
_output_shapes
:*
valueB:
���������
l

Reshape_60Reshapebeta1_power/readReshape_60/shape*
Tshape0*
T0*
_output_shapes
:
c
Reshape_61/shapeConst*
valueB:
���������*
dtype0*
_output_shapes
:
l

Reshape_61Reshapebeta2_power/readReshape_61/shape*
Tshape0*
_output_shapes
:*
T0
c
Reshape_62/shapeConst*
valueB:
���������*
dtype0*
_output_shapes
:
v

Reshape_62Reshapepi/dense/kernel/Adam/readReshape_62/shape*
T0*
_output_shapes	
:�*
Tshape0
c
Reshape_63/shapeConst*
valueB:
���������*
dtype0*
_output_shapes
:
x

Reshape_63Reshapepi/dense/kernel/Adam_1/readReshape_63/shape*
T0*
Tshape0*
_output_shapes	
:�
c
Reshape_64/shapeConst*
dtype0*
_output_shapes
:*
valueB:
���������
s

Reshape_64Reshapepi/dense/bias/Adam/readReshape_64/shape*
Tshape0*
_output_shapes
:@*
T0
c
Reshape_65/shapeConst*
valueB:
���������*
dtype0*
_output_shapes
:
u

Reshape_65Reshapepi/dense/bias/Adam_1/readReshape_65/shape*
_output_shapes
:@*
Tshape0*
T0
c
Reshape_66/shapeConst*
valueB:
���������*
_output_shapes
:*
dtype0
x

Reshape_66Reshapepi/dense_1/kernel/Adam/readReshape_66/shape*
_output_shapes	
:� *
T0*
Tshape0
c
Reshape_67/shapeConst*
_output_shapes
:*
dtype0*
valueB:
���������
z

Reshape_67Reshapepi/dense_1/kernel/Adam_1/readReshape_67/shape*
_output_shapes	
:� *
Tshape0*
T0
c
Reshape_68/shapeConst*
valueB:
���������*
dtype0*
_output_shapes
:
u

Reshape_68Reshapepi/dense_1/bias/Adam/readReshape_68/shape*
T0*
_output_shapes
:@*
Tshape0
c
Reshape_69/shapeConst*
dtype0*
valueB:
���������*
_output_shapes
:
w

Reshape_69Reshapepi/dense_1/bias/Adam_1/readReshape_69/shape*
Tshape0*
_output_shapes
:@*
T0
c
Reshape_70/shapeConst*
dtype0*
valueB:
���������*
_output_shapes
:
x

Reshape_70Reshapepi/dense_2/kernel/Adam/readReshape_70/shape*
T0*
Tshape0*
_output_shapes	
:�
c
Reshape_71/shapeConst*
_output_shapes
:*
valueB:
���������*
dtype0
z

Reshape_71Reshapepi/dense_2/kernel/Adam_1/readReshape_71/shape*
T0*
Tshape0*
_output_shapes	
:�
c
Reshape_72/shapeConst*
dtype0*
_output_shapes
:*
valueB:
���������
u

Reshape_72Reshapepi/dense_2/bias/Adam/readReshape_72/shape*
T0*
_output_shapes
:*
Tshape0
c
Reshape_73/shapeConst*
dtype0*
_output_shapes
:*
valueB:
���������
w

Reshape_73Reshapepi/dense_2/bias/Adam_1/readReshape_73/shape*
T0*
_output_shapes
:*
Tshape0
c
Reshape_74/shapeConst*
dtype0*
valueB:
���������*
_output_shapes
:
n

Reshape_74Reshapebeta1_power_1/readReshape_74/shape*
Tshape0*
T0*
_output_shapes
:
c
Reshape_75/shapeConst*
_output_shapes
:*
dtype0*
valueB:
���������
n

Reshape_75Reshapebeta2_power_1/readReshape_75/shape*
Tshape0*
T0*
_output_shapes
:
c
Reshape_76/shapeConst*
valueB:
���������*
_output_shapes
:*
dtype0
u

Reshape_76Reshapev/dense/kernel/Adam/readReshape_76/shape*
_output_shapes	
:�*
T0*
Tshape0
c
Reshape_77/shapeConst*
valueB:
���������*
_output_shapes
:*
dtype0
w

Reshape_77Reshapev/dense/kernel/Adam_1/readReshape_77/shape*
T0*
Tshape0*
_output_shapes	
:�
c
Reshape_78/shapeConst*
valueB:
���������*
_output_shapes
:*
dtype0
r

Reshape_78Reshapev/dense/bias/Adam/readReshape_78/shape*
_output_shapes
:@*
Tshape0*
T0
c
Reshape_79/shapeConst*
dtype0*
_output_shapes
:*
valueB:
���������
t

Reshape_79Reshapev/dense/bias/Adam_1/readReshape_79/shape*
Tshape0*
T0*
_output_shapes
:@
c
Reshape_80/shapeConst*
_output_shapes
:*
valueB:
���������*
dtype0
w

Reshape_80Reshapev/dense_1/kernel/Adam/readReshape_80/shape*
Tshape0*
_output_shapes	
:� *
T0
c
Reshape_81/shapeConst*
dtype0*
_output_shapes
:*
valueB:
���������
y

Reshape_81Reshapev/dense_1/kernel/Adam_1/readReshape_81/shape*
T0*
_output_shapes	
:� *
Tshape0
c
Reshape_82/shapeConst*
dtype0*
valueB:
���������*
_output_shapes
:
t

Reshape_82Reshapev/dense_1/bias/Adam/readReshape_82/shape*
Tshape0*
T0*
_output_shapes
:@
c
Reshape_83/shapeConst*
dtype0*
_output_shapes
:*
valueB:
���������
v

Reshape_83Reshapev/dense_1/bias/Adam_1/readReshape_83/shape*
_output_shapes
:@*
T0*
Tshape0
c
Reshape_84/shapeConst*
valueB:
���������*
dtype0*
_output_shapes
:
v

Reshape_84Reshapev/dense_2/kernel/Adam/readReshape_84/shape*
_output_shapes
:@*
Tshape0*
T0
c
Reshape_85/shapeConst*
valueB:
���������*
dtype0*
_output_shapes
:
x

Reshape_85Reshapev/dense_2/kernel/Adam_1/readReshape_85/shape*
_output_shapes
:@*
Tshape0*
T0
c
Reshape_86/shapeConst*
valueB:
���������*
dtype0*
_output_shapes
:
t

Reshape_86Reshapev/dense_2/bias/Adam/readReshape_86/shape*
Tshape0*
T0*
_output_shapes
:
c
Reshape_87/shapeConst*
dtype0*
_output_shapes
:*
valueB:
���������
v

Reshape_87Reshapev/dense_2/bias/Adam_1/readReshape_87/shape*
Tshape0*
_output_shapes
:*
T0
O
concat_4/axisConst*
dtype0*
value	B : *
_output_shapes
: 
�
concat_4ConcatV2
Reshape_48
Reshape_49
Reshape_50
Reshape_51
Reshape_52
Reshape_53
Reshape_54
Reshape_55
Reshape_56
Reshape_57
Reshape_58
Reshape_59
Reshape_60
Reshape_61
Reshape_62
Reshape_63
Reshape_64
Reshape_65
Reshape_66
Reshape_67
Reshape_68
Reshape_69
Reshape_70
Reshape_71
Reshape_72
Reshape_73
Reshape_74
Reshape_75
Reshape_76
Reshape_77
Reshape_78
Reshape_79
Reshape_80
Reshape_81
Reshape_82
Reshape_83
Reshape_84
Reshape_85
Reshape_86
Reshape_87concat_4/axis*
N(*
_output_shapes

:��*
T0*

Tidx0
h
PyFunc_4PyFuncconcat_4*
token
pyfunc_4*
Tin
2*
Tout
2*
_output_shapes
:
�
Const_9Const*
dtype0*
_output_shapes
:(*�
value�B�("��  @      @   �      �  @      @   @            �  �  @   @         @   @   �   �               �  �  @   @         @   @   @   @         
S
split_4/split_dimConst*
_output_shapes
: *
value	B : *
dtype0
�
split_4SplitVPyFunc_4Const_9split_4/split_dim*
T0*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::*

Tlen0*
	num_split(
a
Reshape_88/shapeConst*
dtype0*
_output_shapes
:*
valueB"   @   
g

Reshape_88Reshapesplit_4Reshape_88/shape*
Tshape0*
T0*
_output_shapes

:@
Z
Reshape_89/shapeConst*
valueB:@*
_output_shapes
:*
dtype0
e

Reshape_89Reshape	split_4:1Reshape_89/shape*
T0*
Tshape0*
_output_shapes
:@
a
Reshape_90/shapeConst*
dtype0*
_output_shapes
:*
valueB"@   @   
i

Reshape_90Reshape	split_4:2Reshape_90/shape*
_output_shapes

:@@*
Tshape0*
T0
Z
Reshape_91/shapeConst*
dtype0*
_output_shapes
:*
valueB:@
e

Reshape_91Reshape	split_4:3Reshape_91/shape*
_output_shapes
:@*
T0*
Tshape0
a
Reshape_92/shapeConst*
valueB"@      *
dtype0*
_output_shapes
:
i

Reshape_92Reshape	split_4:4Reshape_92/shape*
T0*
Tshape0*
_output_shapes

:@
Z
Reshape_93/shapeConst*
dtype0*
_output_shapes
:*
valueB:
e

Reshape_93Reshape	split_4:5Reshape_93/shape*
T0*
Tshape0*
_output_shapes
:
a
Reshape_94/shapeConst*
dtype0*
valueB"   @   *
_output_shapes
:
i

Reshape_94Reshape	split_4:6Reshape_94/shape*
Tshape0*
T0*
_output_shapes

:@
Z
Reshape_95/shapeConst*
_output_shapes
:*
dtype0*
valueB:@
e

Reshape_95Reshape	split_4:7Reshape_95/shape*
T0*
_output_shapes
:@*
Tshape0
a
Reshape_96/shapeConst*
valueB"@   @   *
_output_shapes
:*
dtype0
i

Reshape_96Reshape	split_4:8Reshape_96/shape*
Tshape0*
_output_shapes

:@@*
T0
Z
Reshape_97/shapeConst*
_output_shapes
:*
dtype0*
valueB:@
e

Reshape_97Reshape	split_4:9Reshape_97/shape*
T0*
_output_shapes
:@*
Tshape0
a
Reshape_98/shapeConst*
valueB"@      *
dtype0*
_output_shapes
:
j

Reshape_98Reshape
split_4:10Reshape_98/shape*
_output_shapes

:@*
Tshape0*
T0
Z
Reshape_99/shapeConst*
dtype0*
_output_shapes
:*
valueB:
f

Reshape_99Reshape
split_4:11Reshape_99/shape*
T0*
_output_shapes
:*
Tshape0
T
Reshape_100/shapeConst*
dtype0*
valueB *
_output_shapes
: 
d
Reshape_100Reshape
split_4:12Reshape_100/shape*
T0*
Tshape0*
_output_shapes
: 
T
Reshape_101/shapeConst*
valueB *
_output_shapes
: *
dtype0
d
Reshape_101Reshape
split_4:13Reshape_101/shape*
Tshape0*
_output_shapes
: *
T0
b
Reshape_102/shapeConst*
_output_shapes
:*
dtype0*
valueB"   @   
l
Reshape_102Reshape
split_4:14Reshape_102/shape*
Tshape0*
T0*
_output_shapes

:@
b
Reshape_103/shapeConst*
valueB"   @   *
dtype0*
_output_shapes
:
l
Reshape_103Reshape
split_4:15Reshape_103/shape*
T0*
Tshape0*
_output_shapes

:@
[
Reshape_104/shapeConst*
dtype0*
valueB:@*
_output_shapes
:
h
Reshape_104Reshape
split_4:16Reshape_104/shape*
_output_shapes
:@*
Tshape0*
T0
[
Reshape_105/shapeConst*
dtype0*
valueB:@*
_output_shapes
:
h
Reshape_105Reshape
split_4:17Reshape_105/shape*
T0*
_output_shapes
:@*
Tshape0
b
Reshape_106/shapeConst*
dtype0*
valueB"@   @   *
_output_shapes
:
l
Reshape_106Reshape
split_4:18Reshape_106/shape*
_output_shapes

:@@*
T0*
Tshape0
b
Reshape_107/shapeConst*
dtype0*
_output_shapes
:*
valueB"@   @   
l
Reshape_107Reshape
split_4:19Reshape_107/shape*
_output_shapes

:@@*
T0*
Tshape0
[
Reshape_108/shapeConst*
_output_shapes
:*
dtype0*
valueB:@
h
Reshape_108Reshape
split_4:20Reshape_108/shape*
T0*
_output_shapes
:@*
Tshape0
[
Reshape_109/shapeConst*
dtype0*
valueB:@*
_output_shapes
:
h
Reshape_109Reshape
split_4:21Reshape_109/shape*
_output_shapes
:@*
T0*
Tshape0
b
Reshape_110/shapeConst*
valueB"@      *
dtype0*
_output_shapes
:
l
Reshape_110Reshape
split_4:22Reshape_110/shape*
_output_shapes

:@*
Tshape0*
T0
b
Reshape_111/shapeConst*
_output_shapes
:*
dtype0*
valueB"@      
l
Reshape_111Reshape
split_4:23Reshape_111/shape*
_output_shapes

:@*
Tshape0*
T0
[
Reshape_112/shapeConst*
valueB:*
_output_shapes
:*
dtype0
h
Reshape_112Reshape
split_4:24Reshape_112/shape*
T0*
Tshape0*
_output_shapes
:
[
Reshape_113/shapeConst*
valueB:*
_output_shapes
:*
dtype0
h
Reshape_113Reshape
split_4:25Reshape_113/shape*
Tshape0*
_output_shapes
:*
T0
T
Reshape_114/shapeConst*
valueB *
_output_shapes
: *
dtype0
d
Reshape_114Reshape
split_4:26Reshape_114/shape*
_output_shapes
: *
Tshape0*
T0
T
Reshape_115/shapeConst*
_output_shapes
: *
dtype0*
valueB 
d
Reshape_115Reshape
split_4:27Reshape_115/shape*
_output_shapes
: *
T0*
Tshape0
b
Reshape_116/shapeConst*
dtype0*
_output_shapes
:*
valueB"   @   
l
Reshape_116Reshape
split_4:28Reshape_116/shape*
Tshape0*
T0*
_output_shapes

:@
b
Reshape_117/shapeConst*
_output_shapes
:*
valueB"   @   *
dtype0
l
Reshape_117Reshape
split_4:29Reshape_117/shape*
_output_shapes

:@*
Tshape0*
T0
[
Reshape_118/shapeConst*
valueB:@*
dtype0*
_output_shapes
:
h
Reshape_118Reshape
split_4:30Reshape_118/shape*
Tshape0*
T0*
_output_shapes
:@
[
Reshape_119/shapeConst*
valueB:@*
dtype0*
_output_shapes
:
h
Reshape_119Reshape
split_4:31Reshape_119/shape*
_output_shapes
:@*
T0*
Tshape0
b
Reshape_120/shapeConst*
dtype0*
valueB"@   @   *
_output_shapes
:
l
Reshape_120Reshape
split_4:32Reshape_120/shape*
_output_shapes

:@@*
T0*
Tshape0
b
Reshape_121/shapeConst*
_output_shapes
:*
dtype0*
valueB"@   @   
l
Reshape_121Reshape
split_4:33Reshape_121/shape*
_output_shapes

:@@*
Tshape0*
T0
[
Reshape_122/shapeConst*
_output_shapes
:*
dtype0*
valueB:@
h
Reshape_122Reshape
split_4:34Reshape_122/shape*
Tshape0*
T0*
_output_shapes
:@
[
Reshape_123/shapeConst*
_output_shapes
:*
valueB:@*
dtype0
h
Reshape_123Reshape
split_4:35Reshape_123/shape*
_output_shapes
:@*
Tshape0*
T0
b
Reshape_124/shapeConst*
dtype0*
valueB"@      *
_output_shapes
:
l
Reshape_124Reshape
split_4:36Reshape_124/shape*
Tshape0*
T0*
_output_shapes

:@
b
Reshape_125/shapeConst*
valueB"@      *
dtype0*
_output_shapes
:
l
Reshape_125Reshape
split_4:37Reshape_125/shape*
Tshape0*
T0*
_output_shapes

:@
[
Reshape_126/shapeConst*
valueB:*
dtype0*
_output_shapes
:
h
Reshape_126Reshape
split_4:38Reshape_126/shape*
T0*
_output_shapes
:*
Tshape0
[
Reshape_127/shapeConst*
dtype0*
valueB:*
_output_shapes
:
h
Reshape_127Reshape
split_4:39Reshape_127/shape*
_output_shapes
:*
T0*
Tshape0
�
	Assign_12Assignpi/dense/kernel
Reshape_88*
_output_shapes

:@*
validate_shape(*
T0*"
_class
loc:@pi/dense/kernel*
use_locking(
�
	Assign_13Assignpi/dense/bias
Reshape_89*
use_locking(*
validate_shape(* 
_class
loc:@pi/dense/bias*
_output_shapes
:@*
T0
�
	Assign_14Assignpi/dense_1/kernel
Reshape_90*
use_locking(*
_output_shapes

:@@*$
_class
loc:@pi/dense_1/kernel*
T0*
validate_shape(
�
	Assign_15Assignpi/dense_1/bias
Reshape_91*
T0*
_output_shapes
:@*
validate_shape(*"
_class
loc:@pi/dense_1/bias*
use_locking(
�
	Assign_16Assignpi/dense_2/kernel
Reshape_92*
_output_shapes

:@*
T0*
use_locking(*$
_class
loc:@pi/dense_2/kernel*
validate_shape(
�
	Assign_17Assignpi/dense_2/bias
Reshape_93*
T0*
validate_shape(*
use_locking(*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:
�
	Assign_18Assignv/dense/kernel
Reshape_94*
use_locking(*!
_class
loc:@v/dense/kernel*
T0*
_output_shapes

:@*
validate_shape(
�
	Assign_19Assignv/dense/bias
Reshape_95*
use_locking(*
_output_shapes
:@*
validate_shape(*
T0*
_class
loc:@v/dense/bias
�
	Assign_20Assignv/dense_1/kernel
Reshape_96*
T0*
validate_shape(*
use_locking(*#
_class
loc:@v/dense_1/kernel*
_output_shapes

:@@
�
	Assign_21Assignv/dense_1/bias
Reshape_97*
T0*
use_locking(*
_output_shapes
:@*
validate_shape(*!
_class
loc:@v/dense_1/bias
�
	Assign_22Assignv/dense_2/kernel
Reshape_98*
validate_shape(*
_output_shapes

:@*#
_class
loc:@v/dense_2/kernel*
use_locking(*
T0
�
	Assign_23Assignv/dense_2/bias
Reshape_99*
T0*
_output_shapes
:*
validate_shape(*!
_class
loc:@v/dense_2/bias*
use_locking(
�
	Assign_24Assignbeta1_powerReshape_100*
T0*
use_locking(*
_output_shapes
: *
validate_shape(* 
_class
loc:@pi/dense/bias
�
	Assign_25Assignbeta2_powerReshape_101* 
_class
loc:@pi/dense/bias*
use_locking(*
_output_shapes
: *
validate_shape(*
T0
�
	Assign_26Assignpi/dense/kernel/AdamReshape_102*
validate_shape(*"
_class
loc:@pi/dense/kernel*
use_locking(*
_output_shapes

:@*
T0
�
	Assign_27Assignpi/dense/kernel/Adam_1Reshape_103*"
_class
loc:@pi/dense/kernel*
T0*
validate_shape(*
_output_shapes

:@*
use_locking(
�
	Assign_28Assignpi/dense/bias/AdamReshape_104*
_output_shapes
:@*
validate_shape(*
T0* 
_class
loc:@pi/dense/bias*
use_locking(
�
	Assign_29Assignpi/dense/bias/Adam_1Reshape_105*
use_locking(*
T0*
validate_shape(*
_output_shapes
:@* 
_class
loc:@pi/dense/bias
�
	Assign_30Assignpi/dense_1/kernel/AdamReshape_106*
T0*
use_locking(*$
_class
loc:@pi/dense_1/kernel*
_output_shapes

:@@*
validate_shape(
�
	Assign_31Assignpi/dense_1/kernel/Adam_1Reshape_107*
_output_shapes

:@@*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel*
validate_shape(
�
	Assign_32Assignpi/dense_1/bias/AdamReshape_108*
_output_shapes
:@*
T0*
use_locking(*"
_class
loc:@pi/dense_1/bias*
validate_shape(
�
	Assign_33Assignpi/dense_1/bias/Adam_1Reshape_109*
validate_shape(*
_output_shapes
:@*
T0*"
_class
loc:@pi/dense_1/bias*
use_locking(
�
	Assign_34Assignpi/dense_2/kernel/AdamReshape_110*
validate_shape(*
T0*$
_class
loc:@pi/dense_2/kernel*
_output_shapes

:@*
use_locking(
�
	Assign_35Assignpi/dense_2/kernel/Adam_1Reshape_111*
use_locking(*
_output_shapes

:@*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
T0
�
	Assign_36Assignpi/dense_2/bias/AdamReshape_112*
T0*"
_class
loc:@pi/dense_2/bias*
use_locking(*
_output_shapes
:*
validate_shape(
�
	Assign_37Assignpi/dense_2/bias/Adam_1Reshape_113*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:*
T0*
use_locking(*
validate_shape(
�
	Assign_38Assignbeta1_power_1Reshape_114*
T0*
_class
loc:@v/dense/bias*
_output_shapes
: *
use_locking(*
validate_shape(
�
	Assign_39Assignbeta2_power_1Reshape_115*
validate_shape(*
use_locking(*
_output_shapes
: *
_class
loc:@v/dense/bias*
T0
�
	Assign_40Assignv/dense/kernel/AdamReshape_116*
_output_shapes

:@*
validate_shape(*
T0*!
_class
loc:@v/dense/kernel*
use_locking(
�
	Assign_41Assignv/dense/kernel/Adam_1Reshape_117*
use_locking(*
T0*!
_class
loc:@v/dense/kernel*
_output_shapes

:@*
validate_shape(
�
	Assign_42Assignv/dense/bias/AdamReshape_118*
_class
loc:@v/dense/bias*
validate_shape(*
T0*
_output_shapes
:@*
use_locking(
�
	Assign_43Assignv/dense/bias/Adam_1Reshape_119*
validate_shape(*
_output_shapes
:@*
T0*
_class
loc:@v/dense/bias*
use_locking(
�
	Assign_44Assignv/dense_1/kernel/AdamReshape_120*#
_class
loc:@v/dense_1/kernel*
validate_shape(*
T0*
use_locking(*
_output_shapes

:@@
�
	Assign_45Assignv/dense_1/kernel/Adam_1Reshape_121*
_output_shapes

:@@*
T0*#
_class
loc:@v/dense_1/kernel*
use_locking(*
validate_shape(
�
	Assign_46Assignv/dense_1/bias/AdamReshape_122*
use_locking(*!
_class
loc:@v/dense_1/bias*
T0*
validate_shape(*
_output_shapes
:@
�
	Assign_47Assignv/dense_1/bias/Adam_1Reshape_123*
_output_shapes
:@*
validate_shape(*
T0*!
_class
loc:@v/dense_1/bias*
use_locking(
�
	Assign_48Assignv/dense_2/kernel/AdamReshape_124*
validate_shape(*
use_locking(*#
_class
loc:@v/dense_2/kernel*
T0*
_output_shapes

:@
�
	Assign_49Assignv/dense_2/kernel/Adam_1Reshape_125*
_output_shapes

:@*
T0*
use_locking(*
validate_shape(*#
_class
loc:@v/dense_2/kernel
�
	Assign_50Assignv/dense_2/bias/AdamReshape_126*
use_locking(*
validate_shape(*
T0*
_output_shapes
:*!
_class
loc:@v/dense_2/bias
�
	Assign_51Assignv/dense_2/bias/Adam_1Reshape_127*
validate_shape(*
T0*
use_locking(*!
_class
loc:@v/dense_2/bias*
_output_shapes
:
�
group_deps_4NoOp
^Assign_12
^Assign_13
^Assign_14
^Assign_15
^Assign_16
^Assign_17
^Assign_18
^Assign_19
^Assign_20
^Assign_21
^Assign_22
^Assign_23
^Assign_24
^Assign_25
^Assign_26
^Assign_27
^Assign_28
^Assign_29
^Assign_30
^Assign_31
^Assign_32
^Assign_33
^Assign_34
^Assign_35
^Assign_36
^Assign_37
^Assign_38
^Assign_39
^Assign_40
^Assign_41
^Assign_42
^Assign_43
^Assign_44
^Assign_45
^Assign_46
^Assign_47
^Assign_48
^Assign_49
^Assign_50
^Assign_51
Y
save/filename/inputConst*
_output_shapes
: *
valueB Bmodel*
dtype0
n
save/filenamePlaceholderWithDefaultsave/filename/input*
dtype0*
shape: *
_output_shapes
: 
e

save/ConstPlaceholderWithDefaultsave/filename*
_output_shapes
: *
shape: *
dtype0
�
save/StringJoin/inputs_1Const*
dtype0*
_output_shapes
: *<
value3B1 B+_temp_bff35ad25c5f430fa41597af754d2874/part
u
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
Q
save/num_shardsConst*
dtype0*
_output_shapes
: *
value	B :
\
save/ShardedFilename/shardConst*
dtype0*
value	B : *
_output_shapes
: 
}
save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards*
_output_shapes
: 
�
save/SaveV2/tensor_namesConst*
dtype0*
_output_shapes
:(*�
value�B�(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1
�
save/SaveV2/shape_and_slicesConst*
_output_shapes
:(*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
�
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1v/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1v/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1v/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1v/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1v/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1v/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1*6
dtypes,
*2(
�
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2*
T0*'
_class
loc:@save/ShardedFilename*
_output_shapes
: 
�
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency*
N*
_output_shapes
:*
T0*

axis 
}
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const*
delete_old_dirs(
z
save/IdentityIdentity
save/Const^save/MergeV2Checkpoints^save/control_dependency*
_output_shapes
: *
T0
�
save/RestoreV2/tensor_namesConst*�
value�B�(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:(
�
save/RestoreV2/shape_and_slicesConst*
_output_shapes
:(*
dtype0*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
�
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::*6
dtypes,
*2(
�
save/AssignAssignbeta1_powersave/RestoreV2*
T0*
_output_shapes
: *
use_locking(*
validate_shape(* 
_class
loc:@pi/dense/bias
�
save/Assign_1Assignbeta1_power_1save/RestoreV2:1*
T0*
_output_shapes
: *
_class
loc:@v/dense/bias*
use_locking(*
validate_shape(
�
save/Assign_2Assignbeta2_powersave/RestoreV2:2*
validate_shape(*
_output_shapes
: *
use_locking(*
T0* 
_class
loc:@pi/dense/bias
�
save/Assign_3Assignbeta2_power_1save/RestoreV2:3*
_output_shapes
: *
_class
loc:@v/dense/bias*
validate_shape(*
T0*
use_locking(
�
save/Assign_4Assignpi/dense/biassave/RestoreV2:4*
_output_shapes
:@* 
_class
loc:@pi/dense/bias*
use_locking(*
T0*
validate_shape(
�
save/Assign_5Assignpi/dense/bias/Adamsave/RestoreV2:5*
validate_shape(*
use_locking(* 
_class
loc:@pi/dense/bias*
T0*
_output_shapes
:@
�
save/Assign_6Assignpi/dense/bias/Adam_1save/RestoreV2:6*
validate_shape(*
use_locking(*
_output_shapes
:@*
T0* 
_class
loc:@pi/dense/bias
�
save/Assign_7Assignpi/dense/kernelsave/RestoreV2:7*
T0*
_output_shapes

:@*
use_locking(*
validate_shape(*"
_class
loc:@pi/dense/kernel
�
save/Assign_8Assignpi/dense/kernel/Adamsave/RestoreV2:8*
_output_shapes

:@*"
_class
loc:@pi/dense/kernel*
validate_shape(*
T0*
use_locking(
�
save/Assign_9Assignpi/dense/kernel/Adam_1save/RestoreV2:9*
_output_shapes

:@*"
_class
loc:@pi/dense/kernel*
use_locking(*
validate_shape(*
T0
�
save/Assign_10Assignpi/dense_1/biassave/RestoreV2:10*
use_locking(*
T0*
_output_shapes
:@*"
_class
loc:@pi/dense_1/bias*
validate_shape(
�
save/Assign_11Assignpi/dense_1/bias/Adamsave/RestoreV2:11*
use_locking(*
validate_shape(*
_output_shapes
:@*"
_class
loc:@pi/dense_1/bias*
T0
�
save/Assign_12Assignpi/dense_1/bias/Adam_1save/RestoreV2:12*"
_class
loc:@pi/dense_1/bias*
use_locking(*
_output_shapes
:@*
T0*
validate_shape(
�
save/Assign_13Assignpi/dense_1/kernelsave/RestoreV2:13*
validate_shape(*
use_locking(*$
_class
loc:@pi/dense_1/kernel*
_output_shapes

:@@*
T0
�
save/Assign_14Assignpi/dense_1/kernel/Adamsave/RestoreV2:14*
_output_shapes

:@@*$
_class
loc:@pi/dense_1/kernel*
T0*
use_locking(*
validate_shape(
�
save/Assign_15Assignpi/dense_1/kernel/Adam_1save/RestoreV2:15*
T0*
_output_shapes

:@@*
validate_shape(*$
_class
loc:@pi/dense_1/kernel*
use_locking(
�
save/Assign_16Assignpi/dense_2/biassave/RestoreV2:16*
T0*
use_locking(*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
_output_shapes
:
�
save/Assign_17Assignpi/dense_2/bias/Adamsave/RestoreV2:17*
use_locking(*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
T0*
_output_shapes
:
�
save/Assign_18Assignpi/dense_2/bias/Adam_1save/RestoreV2:18*
_output_shapes
:*"
_class
loc:@pi/dense_2/bias*
use_locking(*
T0*
validate_shape(
�
save/Assign_19Assignpi/dense_2/kernelsave/RestoreV2:19*
use_locking(*$
_class
loc:@pi/dense_2/kernel*
T0*
_output_shapes

:@*
validate_shape(
�
save/Assign_20Assignpi/dense_2/kernel/Adamsave/RestoreV2:20*
use_locking(*
T0*
validate_shape(*$
_class
loc:@pi/dense_2/kernel*
_output_shapes

:@
�
save/Assign_21Assignpi/dense_2/kernel/Adam_1save/RestoreV2:21*$
_class
loc:@pi/dense_2/kernel*
use_locking(*
T0*
_output_shapes

:@*
validate_shape(
�
save/Assign_22Assignv/dense/biassave/RestoreV2:22*
_output_shapes
:@*
T0*
use_locking(*
_class
loc:@v/dense/bias*
validate_shape(
�
save/Assign_23Assignv/dense/bias/Adamsave/RestoreV2:23*
_class
loc:@v/dense/bias*
use_locking(*
T0*
_output_shapes
:@*
validate_shape(
�
save/Assign_24Assignv/dense/bias/Adam_1save/RestoreV2:24*
use_locking(*
_output_shapes
:@*
validate_shape(*
_class
loc:@v/dense/bias*
T0
�
save/Assign_25Assignv/dense/kernelsave/RestoreV2:25*
use_locking(*
_output_shapes

:@*
T0*!
_class
loc:@v/dense/kernel*
validate_shape(
�
save/Assign_26Assignv/dense/kernel/Adamsave/RestoreV2:26*
validate_shape(*
T0*
use_locking(*!
_class
loc:@v/dense/kernel*
_output_shapes

:@
�
save/Assign_27Assignv/dense/kernel/Adam_1save/RestoreV2:27*!
_class
loc:@v/dense/kernel*
_output_shapes

:@*
validate_shape(*
use_locking(*
T0
�
save/Assign_28Assignv/dense_1/biassave/RestoreV2:28*
use_locking(*
_output_shapes
:@*
T0*!
_class
loc:@v/dense_1/bias*
validate_shape(
�
save/Assign_29Assignv/dense_1/bias/Adamsave/RestoreV2:29*
use_locking(*
_output_shapes
:@*
validate_shape(*!
_class
loc:@v/dense_1/bias*
T0
�
save/Assign_30Assignv/dense_1/bias/Adam_1save/RestoreV2:30*
_output_shapes
:@*
validate_shape(*
use_locking(*
T0*!
_class
loc:@v/dense_1/bias
�
save/Assign_31Assignv/dense_1/kernelsave/RestoreV2:31*
T0*
_output_shapes

:@@*
validate_shape(*#
_class
loc:@v/dense_1/kernel*
use_locking(
�
save/Assign_32Assignv/dense_1/kernel/Adamsave/RestoreV2:32*#
_class
loc:@v/dense_1/kernel*
use_locking(*
validate_shape(*
T0*
_output_shapes

:@@
�
save/Assign_33Assignv/dense_1/kernel/Adam_1save/RestoreV2:33*
use_locking(*
T0*#
_class
loc:@v/dense_1/kernel*
_output_shapes

:@@*
validate_shape(
�
save/Assign_34Assignv/dense_2/biassave/RestoreV2:34*
validate_shape(*
T0*!
_class
loc:@v/dense_2/bias*
_output_shapes
:*
use_locking(
�
save/Assign_35Assignv/dense_2/bias/Adamsave/RestoreV2:35*
T0*
_output_shapes
:*
validate_shape(*!
_class
loc:@v/dense_2/bias*
use_locking(
�
save/Assign_36Assignv/dense_2/bias/Adam_1save/RestoreV2:36*
use_locking(*
_output_shapes
:*!
_class
loc:@v/dense_2/bias*
validate_shape(*
T0
�
save/Assign_37Assignv/dense_2/kernelsave/RestoreV2:37*
T0*
_output_shapes

:@*
use_locking(*#
_class
loc:@v/dense_2/kernel*
validate_shape(
�
save/Assign_38Assignv/dense_2/kernel/Adamsave/RestoreV2:38*
use_locking(*
T0*
_output_shapes

:@*
validate_shape(*#
_class
loc:@v/dense_2/kernel
�
save/Assign_39Assignv/dense_2/kernel/Adam_1save/RestoreV2:39*
T0*#
_class
loc:@v/dense_2/kernel*
validate_shape(*
_output_shapes

:@*
use_locking(
�
save/restore_shardNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_2^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25^save/Assign_26^save/Assign_27^save/Assign_28^save/Assign_29^save/Assign_3^save/Assign_30^save/Assign_31^save/Assign_32^save/Assign_33^save/Assign_34^save/Assign_35^save/Assign_36^save/Assign_37^save/Assign_38^save/Assign_39^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9
-
save/restore_allNoOp^save/restore_shard
[
save_1/filename/inputConst*
_output_shapes
: *
valueB Bmodel*
dtype0
r
save_1/filenamePlaceholderWithDefaultsave_1/filename/input*
dtype0*
shape: *
_output_shapes
: 
i
save_1/ConstPlaceholderWithDefaultsave_1/filename*
_output_shapes
: *
shape: *
dtype0
�
save_1/StringJoin/inputs_1Const*<
value3B1 B+_temp_a5b805c678424e1784c79438f3ddf057/part*
dtype0*
_output_shapes
: 
{
save_1/StringJoin
StringJoinsave_1/Constsave_1/StringJoin/inputs_1*
_output_shapes
: *
	separator *
N
S
save_1/num_shardsConst*
_output_shapes
: *
value	B :*
dtype0
^
save_1/ShardedFilename/shardConst*
_output_shapes
: *
dtype0*
value	B : 
�
save_1/ShardedFilenameShardedFilenamesave_1/StringJoinsave_1/ShardedFilename/shardsave_1/num_shards*
_output_shapes
: 
�
save_1/SaveV2/tensor_namesConst*
_output_shapes
:(*�
value�B�(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
dtype0
�
save_1/SaveV2/shape_and_slicesConst*
_output_shapes
:(*
dtype0*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
�
save_1/SaveV2SaveV2save_1/ShardedFilenamesave_1/SaveV2/tensor_namessave_1/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1v/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1v/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1v/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1v/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1v/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1v/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1*6
dtypes,
*2(
�
save_1/control_dependencyIdentitysave_1/ShardedFilename^save_1/SaveV2*)
_class
loc:@save_1/ShardedFilename*
T0*
_output_shapes
: 
�
-save_1/MergeV2Checkpoints/checkpoint_prefixesPacksave_1/ShardedFilename^save_1/control_dependency*
_output_shapes
:*

axis *
N*
T0
�
save_1/MergeV2CheckpointsMergeV2Checkpoints-save_1/MergeV2Checkpoints/checkpoint_prefixessave_1/Const*
delete_old_dirs(
�
save_1/IdentityIdentitysave_1/Const^save_1/MergeV2Checkpoints^save_1/control_dependency*
T0*
_output_shapes
: 
�
save_1/RestoreV2/tensor_namesConst*
_output_shapes
:(*
dtype0*�
value�B�(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1
�
!save_1/RestoreV2/shape_and_slicesConst*
_output_shapes
:(*
dtype0*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
�
save_1/RestoreV2	RestoreV2save_1/Constsave_1/RestoreV2/tensor_names!save_1/RestoreV2/shape_and_slices*6
dtypes,
*2(*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::
�
save_1/AssignAssignbeta1_powersave_1/RestoreV2*
validate_shape(* 
_class
loc:@pi/dense/bias*
T0*
use_locking(*
_output_shapes
: 
�
save_1/Assign_1Assignbeta1_power_1save_1/RestoreV2:1*
T0*
validate_shape(*
_class
loc:@v/dense/bias*
use_locking(*
_output_shapes
: 
�
save_1/Assign_2Assignbeta2_powersave_1/RestoreV2:2*
T0*
validate_shape(*
_output_shapes
: *
use_locking(* 
_class
loc:@pi/dense/bias
�
save_1/Assign_3Assignbeta2_power_1save_1/RestoreV2:3*
validate_shape(*
T0*
use_locking(*
_output_shapes
: *
_class
loc:@v/dense/bias
�
save_1/Assign_4Assignpi/dense/biassave_1/RestoreV2:4*
T0* 
_class
loc:@pi/dense/bias*
use_locking(*
_output_shapes
:@*
validate_shape(
�
save_1/Assign_5Assignpi/dense/bias/Adamsave_1/RestoreV2:5* 
_class
loc:@pi/dense/bias*
_output_shapes
:@*
use_locking(*
validate_shape(*
T0
�
save_1/Assign_6Assignpi/dense/bias/Adam_1save_1/RestoreV2:6*
T0*
use_locking(*
validate_shape(*
_output_shapes
:@* 
_class
loc:@pi/dense/bias
�
save_1/Assign_7Assignpi/dense/kernelsave_1/RestoreV2:7*"
_class
loc:@pi/dense/kernel*
use_locking(*
_output_shapes

:@*
T0*
validate_shape(
�
save_1/Assign_8Assignpi/dense/kernel/Adamsave_1/RestoreV2:8*
_output_shapes

:@*
validate_shape(*
T0*
use_locking(*"
_class
loc:@pi/dense/kernel
�
save_1/Assign_9Assignpi/dense/kernel/Adam_1save_1/RestoreV2:9*"
_class
loc:@pi/dense/kernel*
use_locking(*
T0*
validate_shape(*
_output_shapes

:@
�
save_1/Assign_10Assignpi/dense_1/biassave_1/RestoreV2:10*
_output_shapes
:@*
validate_shape(*
use_locking(*
T0*"
_class
loc:@pi/dense_1/bias
�
save_1/Assign_11Assignpi/dense_1/bias/Adamsave_1/RestoreV2:11*
use_locking(*
validate_shape(*
_output_shapes
:@*"
_class
loc:@pi/dense_1/bias*
T0
�
save_1/Assign_12Assignpi/dense_1/bias/Adam_1save_1/RestoreV2:12*"
_class
loc:@pi/dense_1/bias*
T0*
validate_shape(*
_output_shapes
:@*
use_locking(
�
save_1/Assign_13Assignpi/dense_1/kernelsave_1/RestoreV2:13*
_output_shapes

:@@*
use_locking(*$
_class
loc:@pi/dense_1/kernel*
T0*
validate_shape(
�
save_1/Assign_14Assignpi/dense_1/kernel/Adamsave_1/RestoreV2:14*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
�
save_1/Assign_15Assignpi/dense_1/kernel/Adam_1save_1/RestoreV2:15*$
_class
loc:@pi/dense_1/kernel*
_output_shapes

:@@*
validate_shape(*
use_locking(*
T0
�
save_1/Assign_16Assignpi/dense_2/biassave_1/RestoreV2:16*"
_class
loc:@pi/dense_2/bias*
T0*
_output_shapes
:*
use_locking(*
validate_shape(
�
save_1/Assign_17Assignpi/dense_2/bias/Adamsave_1/RestoreV2:17*
validate_shape(*
_output_shapes
:*
T0*"
_class
loc:@pi/dense_2/bias*
use_locking(
�
save_1/Assign_18Assignpi/dense_2/bias/Adam_1save_1/RestoreV2:18*
validate_shape(*
_output_shapes
:*"
_class
loc:@pi/dense_2/bias*
T0*
use_locking(
�
save_1/Assign_19Assignpi/dense_2/kernelsave_1/RestoreV2:19*$
_class
loc:@pi/dense_2/kernel*
use_locking(*
_output_shapes

:@*
T0*
validate_shape(
�
save_1/Assign_20Assignpi/dense_2/kernel/Adamsave_1/RestoreV2:20*
use_locking(*
_output_shapes

:@*
validate_shape(*$
_class
loc:@pi/dense_2/kernel*
T0
�
save_1/Assign_21Assignpi/dense_2/kernel/Adam_1save_1/RestoreV2:21*$
_class
loc:@pi/dense_2/kernel*
use_locking(*
_output_shapes

:@*
T0*
validate_shape(
�
save_1/Assign_22Assignv/dense/biassave_1/RestoreV2:22*
T0*
_output_shapes
:@*
use_locking(*
_class
loc:@v/dense/bias*
validate_shape(
�
save_1/Assign_23Assignv/dense/bias/Adamsave_1/RestoreV2:23*
use_locking(*
validate_shape(*
_output_shapes
:@*
_class
loc:@v/dense/bias*
T0
�
save_1/Assign_24Assignv/dense/bias/Adam_1save_1/RestoreV2:24*
T0*
validate_shape(*
use_locking(*
_class
loc:@v/dense/bias*
_output_shapes
:@
�
save_1/Assign_25Assignv/dense/kernelsave_1/RestoreV2:25*
T0*
use_locking(*!
_class
loc:@v/dense/kernel*
validate_shape(*
_output_shapes

:@
�
save_1/Assign_26Assignv/dense/kernel/Adamsave_1/RestoreV2:26*
validate_shape(*
T0*
_output_shapes

:@*
use_locking(*!
_class
loc:@v/dense/kernel
�
save_1/Assign_27Assignv/dense/kernel/Adam_1save_1/RestoreV2:27*
_output_shapes

:@*
use_locking(*
T0*
validate_shape(*!
_class
loc:@v/dense/kernel
�
save_1/Assign_28Assignv/dense_1/biassave_1/RestoreV2:28*!
_class
loc:@v/dense_1/bias*
use_locking(*
validate_shape(*
T0*
_output_shapes
:@
�
save_1/Assign_29Assignv/dense_1/bias/Adamsave_1/RestoreV2:29*
_output_shapes
:@*
validate_shape(*
use_locking(*
T0*!
_class
loc:@v/dense_1/bias
�
save_1/Assign_30Assignv/dense_1/bias/Adam_1save_1/RestoreV2:30*!
_class
loc:@v/dense_1/bias*
use_locking(*
T0*
validate_shape(*
_output_shapes
:@
�
save_1/Assign_31Assignv/dense_1/kernelsave_1/RestoreV2:31*
T0*
_output_shapes

:@@*#
_class
loc:@v/dense_1/kernel*
use_locking(*
validate_shape(
�
save_1/Assign_32Assignv/dense_1/kernel/Adamsave_1/RestoreV2:32*
T0*#
_class
loc:@v/dense_1/kernel*
validate_shape(*
use_locking(*
_output_shapes

:@@
�
save_1/Assign_33Assignv/dense_1/kernel/Adam_1save_1/RestoreV2:33*
_output_shapes

:@@*
validate_shape(*#
_class
loc:@v/dense_1/kernel*
T0*
use_locking(
�
save_1/Assign_34Assignv/dense_2/biassave_1/RestoreV2:34*
T0*
use_locking(*
validate_shape(*
_output_shapes
:*!
_class
loc:@v/dense_2/bias
�
save_1/Assign_35Assignv/dense_2/bias/Adamsave_1/RestoreV2:35*
T0*
use_locking(*
_output_shapes
:*
validate_shape(*!
_class
loc:@v/dense_2/bias
�
save_1/Assign_36Assignv/dense_2/bias/Adam_1save_1/RestoreV2:36*
T0*
_output_shapes
:*
use_locking(*
validate_shape(*!
_class
loc:@v/dense_2/bias
�
save_1/Assign_37Assignv/dense_2/kernelsave_1/RestoreV2:37*
validate_shape(*#
_class
loc:@v/dense_2/kernel*
use_locking(*
_output_shapes

:@*
T0
�
save_1/Assign_38Assignv/dense_2/kernel/Adamsave_1/RestoreV2:38*
T0*
use_locking(*#
_class
loc:@v/dense_2/kernel*
_output_shapes

:@*
validate_shape(
�
save_1/Assign_39Assignv/dense_2/kernel/Adam_1save_1/RestoreV2:39*
validate_shape(*#
_class
loc:@v/dense_2/kernel*
_output_shapes

:@*
use_locking(*
T0
�
save_1/restore_shardNoOp^save_1/Assign^save_1/Assign_1^save_1/Assign_10^save_1/Assign_11^save_1/Assign_12^save_1/Assign_13^save_1/Assign_14^save_1/Assign_15^save_1/Assign_16^save_1/Assign_17^save_1/Assign_18^save_1/Assign_19^save_1/Assign_2^save_1/Assign_20^save_1/Assign_21^save_1/Assign_22^save_1/Assign_23^save_1/Assign_24^save_1/Assign_25^save_1/Assign_26^save_1/Assign_27^save_1/Assign_28^save_1/Assign_29^save_1/Assign_3^save_1/Assign_30^save_1/Assign_31^save_1/Assign_32^save_1/Assign_33^save_1/Assign_34^save_1/Assign_35^save_1/Assign_36^save_1/Assign_37^save_1/Assign_38^save_1/Assign_39^save_1/Assign_4^save_1/Assign_5^save_1/Assign_6^save_1/Assign_7^save_1/Assign_8^save_1/Assign_9
1
save_1/restore_allNoOp^save_1/restore_shard
[
save_2/filename/inputConst*
dtype0*
valueB Bmodel*
_output_shapes
: 
r
save_2/filenamePlaceholderWithDefaultsave_2/filename/input*
_output_shapes
: *
shape: *
dtype0
i
save_2/ConstPlaceholderWithDefaultsave_2/filename*
_output_shapes
: *
shape: *
dtype0
�
save_2/StringJoin/inputs_1Const*
dtype0*<
value3B1 B+_temp_93bf199e2b62456f8fec395bfb3d0107/part*
_output_shapes
: 
{
save_2/StringJoin
StringJoinsave_2/Constsave_2/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
S
save_2/num_shardsConst*
_output_shapes
: *
value	B :*
dtype0
^
save_2/ShardedFilename/shardConst*
dtype0*
_output_shapes
: *
value	B : 
�
save_2/ShardedFilenameShardedFilenamesave_2/StringJoinsave_2/ShardedFilename/shardsave_2/num_shards*
_output_shapes
: 
�
save_2/SaveV2/tensor_namesConst*
_output_shapes
:(*
dtype0*�
value�B�(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1
�
save_2/SaveV2/shape_and_slicesConst*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:(*
dtype0
�
save_2/SaveV2SaveV2save_2/ShardedFilenamesave_2/SaveV2/tensor_namessave_2/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1v/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1v/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1v/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1v/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1v/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1v/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1*6
dtypes,
*2(
�
save_2/control_dependencyIdentitysave_2/ShardedFilename^save_2/SaveV2*
_output_shapes
: *)
_class
loc:@save_2/ShardedFilename*
T0
�
-save_2/MergeV2Checkpoints/checkpoint_prefixesPacksave_2/ShardedFilename^save_2/control_dependency*
_output_shapes
:*
N*
T0*

axis 
�
save_2/MergeV2CheckpointsMergeV2Checkpoints-save_2/MergeV2Checkpoints/checkpoint_prefixessave_2/Const*
delete_old_dirs(
�
save_2/IdentityIdentitysave_2/Const^save_2/MergeV2Checkpoints^save_2/control_dependency*
_output_shapes
: *
T0
�
save_2/RestoreV2/tensor_namesConst*�
value�B�(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
_output_shapes
:(*
dtype0
�
!save_2/RestoreV2/shape_and_slicesConst*
dtype0*
_output_shapes
:(*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
�
save_2/RestoreV2	RestoreV2save_2/Constsave_2/RestoreV2/tensor_names!save_2/RestoreV2/shape_and_slices*6
dtypes,
*2(*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::
�
save_2/AssignAssignbeta1_powersave_2/RestoreV2*
_output_shapes
: *
validate_shape(* 
_class
loc:@pi/dense/bias*
use_locking(*
T0
�
save_2/Assign_1Assignbeta1_power_1save_2/RestoreV2:1*
use_locking(*
T0*
validate_shape(*
_output_shapes
: *
_class
loc:@v/dense/bias
�
save_2/Assign_2Assignbeta2_powersave_2/RestoreV2:2* 
_class
loc:@pi/dense/bias*
_output_shapes
: *
validate_shape(*
T0*
use_locking(
�
save_2/Assign_3Assignbeta2_power_1save_2/RestoreV2:3*
T0*
_output_shapes
: *
validate_shape(*
_class
loc:@v/dense/bias*
use_locking(
�
save_2/Assign_4Assignpi/dense/biassave_2/RestoreV2:4*
validate_shape(*
T0*
_output_shapes
:@* 
_class
loc:@pi/dense/bias*
use_locking(
�
save_2/Assign_5Assignpi/dense/bias/Adamsave_2/RestoreV2:5* 
_class
loc:@pi/dense/bias*
_output_shapes
:@*
use_locking(*
validate_shape(*
T0
�
save_2/Assign_6Assignpi/dense/bias/Adam_1save_2/RestoreV2:6*
_output_shapes
:@*
use_locking(* 
_class
loc:@pi/dense/bias*
validate_shape(*
T0
�
save_2/Assign_7Assignpi/dense/kernelsave_2/RestoreV2:7*
_output_shapes

:@*"
_class
loc:@pi/dense/kernel*
validate_shape(*
T0*
use_locking(
�
save_2/Assign_8Assignpi/dense/kernel/Adamsave_2/RestoreV2:8*"
_class
loc:@pi/dense/kernel*
use_locking(*
validate_shape(*
T0*
_output_shapes

:@
�
save_2/Assign_9Assignpi/dense/kernel/Adam_1save_2/RestoreV2:9*
validate_shape(*
T0*
use_locking(*"
_class
loc:@pi/dense/kernel*
_output_shapes

:@
�
save_2/Assign_10Assignpi/dense_1/biassave_2/RestoreV2:10*
use_locking(*
validate_shape(*
_output_shapes
:@*"
_class
loc:@pi/dense_1/bias*
T0
�
save_2/Assign_11Assignpi/dense_1/bias/Adamsave_2/RestoreV2:11*
_output_shapes
:@*
T0*
validate_shape(*
use_locking(*"
_class
loc:@pi/dense_1/bias
�
save_2/Assign_12Assignpi/dense_1/bias/Adam_1save_2/RestoreV2:12*"
_class
loc:@pi/dense_1/bias*
use_locking(*
_output_shapes
:@*
T0*
validate_shape(
�
save_2/Assign_13Assignpi/dense_1/kernelsave_2/RestoreV2:13*
T0*$
_class
loc:@pi/dense_1/kernel*
validate_shape(*
use_locking(*
_output_shapes

:@@
�
save_2/Assign_14Assignpi/dense_1/kernel/Adamsave_2/RestoreV2:14*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
�
save_2/Assign_15Assignpi/dense_1/kernel/Adam_1save_2/RestoreV2:15*$
_class
loc:@pi/dense_1/kernel*
use_locking(*
_output_shapes

:@@*
validate_shape(*
T0
�
save_2/Assign_16Assignpi/dense_2/biassave_2/RestoreV2:16*
use_locking(*
validate_shape(*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:*
T0
�
save_2/Assign_17Assignpi/dense_2/bias/Adamsave_2/RestoreV2:17*
T0*"
_class
loc:@pi/dense_2/bias*
use_locking(*
_output_shapes
:*
validate_shape(
�
save_2/Assign_18Assignpi/dense_2/bias/Adam_1save_2/RestoreV2:18*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:*
validate_shape(*
use_locking(*
T0
�
save_2/Assign_19Assignpi/dense_2/kernelsave_2/RestoreV2:19*$
_class
loc:@pi/dense_2/kernel*
use_locking(*
validate_shape(*
T0*
_output_shapes

:@
�
save_2/Assign_20Assignpi/dense_2/kernel/Adamsave_2/RestoreV2:20*
T0*
validate_shape(*
_output_shapes

:@*$
_class
loc:@pi/dense_2/kernel*
use_locking(
�
save_2/Assign_21Assignpi/dense_2/kernel/Adam_1save_2/RestoreV2:21*$
_class
loc:@pi/dense_2/kernel*
T0*
_output_shapes

:@*
validate_shape(*
use_locking(
�
save_2/Assign_22Assignv/dense/biassave_2/RestoreV2:22*
validate_shape(*
use_locking(*
_class
loc:@v/dense/bias*
_output_shapes
:@*
T0
�
save_2/Assign_23Assignv/dense/bias/Adamsave_2/RestoreV2:23*
T0*
validate_shape(*
_output_shapes
:@*
_class
loc:@v/dense/bias*
use_locking(
�
save_2/Assign_24Assignv/dense/bias/Adam_1save_2/RestoreV2:24*
T0*
use_locking(*
_class
loc:@v/dense/bias*
_output_shapes
:@*
validate_shape(
�
save_2/Assign_25Assignv/dense/kernelsave_2/RestoreV2:25*
validate_shape(*!
_class
loc:@v/dense/kernel*
use_locking(*
T0*
_output_shapes

:@
�
save_2/Assign_26Assignv/dense/kernel/Adamsave_2/RestoreV2:26*
T0*
_output_shapes

:@*!
_class
loc:@v/dense/kernel*
use_locking(*
validate_shape(
�
save_2/Assign_27Assignv/dense/kernel/Adam_1save_2/RestoreV2:27*
use_locking(*
_output_shapes

:@*
T0*!
_class
loc:@v/dense/kernel*
validate_shape(
�
save_2/Assign_28Assignv/dense_1/biassave_2/RestoreV2:28*
T0*
use_locking(*
_output_shapes
:@*!
_class
loc:@v/dense_1/bias*
validate_shape(
�
save_2/Assign_29Assignv/dense_1/bias/Adamsave_2/RestoreV2:29*
validate_shape(*
T0*
_output_shapes
:@*!
_class
loc:@v/dense_1/bias*
use_locking(
�
save_2/Assign_30Assignv/dense_1/bias/Adam_1save_2/RestoreV2:30*
validate_shape(*
_output_shapes
:@*!
_class
loc:@v/dense_1/bias*
use_locking(*
T0
�
save_2/Assign_31Assignv/dense_1/kernelsave_2/RestoreV2:31*
T0*
_output_shapes

:@@*
validate_shape(*
use_locking(*#
_class
loc:@v/dense_1/kernel
�
save_2/Assign_32Assignv/dense_1/kernel/Adamsave_2/RestoreV2:32*
T0*
_output_shapes

:@@*#
_class
loc:@v/dense_1/kernel*
use_locking(*
validate_shape(
�
save_2/Assign_33Assignv/dense_1/kernel/Adam_1save_2/RestoreV2:33*
use_locking(*
_output_shapes

:@@*
validate_shape(*#
_class
loc:@v/dense_1/kernel*
T0
�
save_2/Assign_34Assignv/dense_2/biassave_2/RestoreV2:34*
T0*
use_locking(*!
_class
loc:@v/dense_2/bias*
validate_shape(*
_output_shapes
:
�
save_2/Assign_35Assignv/dense_2/bias/Adamsave_2/RestoreV2:35*
T0*
_output_shapes
:*
validate_shape(*
use_locking(*!
_class
loc:@v/dense_2/bias
�
save_2/Assign_36Assignv/dense_2/bias/Adam_1save_2/RestoreV2:36*
use_locking(*!
_class
loc:@v/dense_2/bias*
validate_shape(*
T0*
_output_shapes
:
�
save_2/Assign_37Assignv/dense_2/kernelsave_2/RestoreV2:37*
_output_shapes

:@*
T0*#
_class
loc:@v/dense_2/kernel*
validate_shape(*
use_locking(
�
save_2/Assign_38Assignv/dense_2/kernel/Adamsave_2/RestoreV2:38*
validate_shape(*
use_locking(*
T0*#
_class
loc:@v/dense_2/kernel*
_output_shapes

:@
�
save_2/Assign_39Assignv/dense_2/kernel/Adam_1save_2/RestoreV2:39*#
_class
loc:@v/dense_2/kernel*
T0*
validate_shape(*
_output_shapes

:@*
use_locking(
�
save_2/restore_shardNoOp^save_2/Assign^save_2/Assign_1^save_2/Assign_10^save_2/Assign_11^save_2/Assign_12^save_2/Assign_13^save_2/Assign_14^save_2/Assign_15^save_2/Assign_16^save_2/Assign_17^save_2/Assign_18^save_2/Assign_19^save_2/Assign_2^save_2/Assign_20^save_2/Assign_21^save_2/Assign_22^save_2/Assign_23^save_2/Assign_24^save_2/Assign_25^save_2/Assign_26^save_2/Assign_27^save_2/Assign_28^save_2/Assign_29^save_2/Assign_3^save_2/Assign_30^save_2/Assign_31^save_2/Assign_32^save_2/Assign_33^save_2/Assign_34^save_2/Assign_35^save_2/Assign_36^save_2/Assign_37^save_2/Assign_38^save_2/Assign_39^save_2/Assign_4^save_2/Assign_5^save_2/Assign_6^save_2/Assign_7^save_2/Assign_8^save_2/Assign_9
1
save_2/restore_allNoOp^save_2/restore_shard
[
save_3/filename/inputConst*
_output_shapes
: *
dtype0*
valueB Bmodel
r
save_3/filenamePlaceholderWithDefaultsave_3/filename/input*
dtype0*
shape: *
_output_shapes
: 
i
save_3/ConstPlaceholderWithDefaultsave_3/filename*
_output_shapes
: *
dtype0*
shape: 
�
save_3/StringJoin/inputs_1Const*<
value3B1 B+_temp_037c5910b2194bc2a276e0eb921c7f2a/part*
_output_shapes
: *
dtype0
{
save_3/StringJoin
StringJoinsave_3/Constsave_3/StringJoin/inputs_1*
	separator *
N*
_output_shapes
: 
S
save_3/num_shardsConst*
_output_shapes
: *
value	B :*
dtype0
^
save_3/ShardedFilename/shardConst*
dtype0*
_output_shapes
: *
value	B : 
�
save_3/ShardedFilenameShardedFilenamesave_3/StringJoinsave_3/ShardedFilename/shardsave_3/num_shards*
_output_shapes
: 
�
save_3/SaveV2/tensor_namesConst*�
value�B�(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
_output_shapes
:(*
dtype0
�
save_3/SaveV2/shape_and_slicesConst*
_output_shapes
:(*
dtype0*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
�
save_3/SaveV2SaveV2save_3/ShardedFilenamesave_3/SaveV2/tensor_namessave_3/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1v/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1v/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1v/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1v/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1v/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1v/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1*6
dtypes,
*2(
�
save_3/control_dependencyIdentitysave_3/ShardedFilename^save_3/SaveV2*)
_class
loc:@save_3/ShardedFilename*
_output_shapes
: *
T0
�
-save_3/MergeV2Checkpoints/checkpoint_prefixesPacksave_3/ShardedFilename^save_3/control_dependency*

axis *
N*
_output_shapes
:*
T0
�
save_3/MergeV2CheckpointsMergeV2Checkpoints-save_3/MergeV2Checkpoints/checkpoint_prefixessave_3/Const*
delete_old_dirs(
�
save_3/IdentityIdentitysave_3/Const^save_3/MergeV2Checkpoints^save_3/control_dependency*
T0*
_output_shapes
: 
�
save_3/RestoreV2/tensor_namesConst*�
value�B�(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
_output_shapes
:(*
dtype0
�
!save_3/RestoreV2/shape_and_slicesConst*
_output_shapes
:(*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
�
save_3/RestoreV2	RestoreV2save_3/Constsave_3/RestoreV2/tensor_names!save_3/RestoreV2/shape_and_slices*6
dtypes,
*2(*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::
�
save_3/AssignAssignbeta1_powersave_3/RestoreV2*
T0*
validate_shape(*
_output_shapes
: *
use_locking(* 
_class
loc:@pi/dense/bias
�
save_3/Assign_1Assignbeta1_power_1save_3/RestoreV2:1*
_class
loc:@v/dense/bias*
use_locking(*
validate_shape(*
_output_shapes
: *
T0
�
save_3/Assign_2Assignbeta2_powersave_3/RestoreV2:2*
validate_shape(*
use_locking(* 
_class
loc:@pi/dense/bias*
T0*
_output_shapes
: 
�
save_3/Assign_3Assignbeta2_power_1save_3/RestoreV2:3*
use_locking(*
T0*
_output_shapes
: *
validate_shape(*
_class
loc:@v/dense/bias
�
save_3/Assign_4Assignpi/dense/biassave_3/RestoreV2:4*
T0*
use_locking(*
_output_shapes
:@* 
_class
loc:@pi/dense/bias*
validate_shape(
�
save_3/Assign_5Assignpi/dense/bias/Adamsave_3/RestoreV2:5*
use_locking(*
_output_shapes
:@*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(
�
save_3/Assign_6Assignpi/dense/bias/Adam_1save_3/RestoreV2:6*
_output_shapes
:@*
T0* 
_class
loc:@pi/dense/bias*
use_locking(*
validate_shape(
�
save_3/Assign_7Assignpi/dense/kernelsave_3/RestoreV2:7*
validate_shape(*"
_class
loc:@pi/dense/kernel*
T0*
use_locking(*
_output_shapes

:@
�
save_3/Assign_8Assignpi/dense/kernel/Adamsave_3/RestoreV2:8*
T0*
_output_shapes

:@*"
_class
loc:@pi/dense/kernel*
use_locking(*
validate_shape(
�
save_3/Assign_9Assignpi/dense/kernel/Adam_1save_3/RestoreV2:9*"
_class
loc:@pi/dense/kernel*
use_locking(*
_output_shapes

:@*
validate_shape(*
T0
�
save_3/Assign_10Assignpi/dense_1/biassave_3/RestoreV2:10*
_output_shapes
:@*
validate_shape(*"
_class
loc:@pi/dense_1/bias*
T0*
use_locking(
�
save_3/Assign_11Assignpi/dense_1/bias/Adamsave_3/RestoreV2:11*
validate_shape(*"
_class
loc:@pi/dense_1/bias*
use_locking(*
T0*
_output_shapes
:@
�
save_3/Assign_12Assignpi/dense_1/bias/Adam_1save_3/RestoreV2:12*
validate_shape(*"
_class
loc:@pi/dense_1/bias*
_output_shapes
:@*
use_locking(*
T0
�
save_3/Assign_13Assignpi/dense_1/kernelsave_3/RestoreV2:13*
validate_shape(*$
_class
loc:@pi/dense_1/kernel*
use_locking(*
T0*
_output_shapes

:@@
�
save_3/Assign_14Assignpi/dense_1/kernel/Adamsave_3/RestoreV2:14*
validate_shape(*
T0*
_output_shapes

:@@*$
_class
loc:@pi/dense_1/kernel*
use_locking(
�
save_3/Assign_15Assignpi/dense_1/kernel/Adam_1save_3/RestoreV2:15*
_output_shapes

:@@*
T0*
validate_shape(*$
_class
loc:@pi/dense_1/kernel*
use_locking(
�
save_3/Assign_16Assignpi/dense_2/biassave_3/RestoreV2:16*
validate_shape(*"
_class
loc:@pi/dense_2/bias*
T0*
use_locking(*
_output_shapes
:
�
save_3/Assign_17Assignpi/dense_2/bias/Adamsave_3/RestoreV2:17*
_output_shapes
:*
T0*
use_locking(*
validate_shape(*"
_class
loc:@pi/dense_2/bias
�
save_3/Assign_18Assignpi/dense_2/bias/Adam_1save_3/RestoreV2:18*
T0*
use_locking(*
_output_shapes
:*"
_class
loc:@pi/dense_2/bias*
validate_shape(
�
save_3/Assign_19Assignpi/dense_2/kernelsave_3/RestoreV2:19*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@*
T0*
use_locking(
�
save_3/Assign_20Assignpi/dense_2/kernel/Adamsave_3/RestoreV2:20*
use_locking(*
validate_shape(*
T0*
_output_shapes

:@*$
_class
loc:@pi/dense_2/kernel
�
save_3/Assign_21Assignpi/dense_2/kernel/Adam_1save_3/RestoreV2:21*
_output_shapes

:@*
validate_shape(*$
_class
loc:@pi/dense_2/kernel*
use_locking(*
T0
�
save_3/Assign_22Assignv/dense/biassave_3/RestoreV2:22*
_class
loc:@v/dense/bias*
use_locking(*
T0*
_output_shapes
:@*
validate_shape(
�
save_3/Assign_23Assignv/dense/bias/Adamsave_3/RestoreV2:23*
_class
loc:@v/dense/bias*
T0*
_output_shapes
:@*
use_locking(*
validate_shape(
�
save_3/Assign_24Assignv/dense/bias/Adam_1save_3/RestoreV2:24*
_output_shapes
:@*
T0*
use_locking(*
_class
loc:@v/dense/bias*
validate_shape(
�
save_3/Assign_25Assignv/dense/kernelsave_3/RestoreV2:25*
use_locking(*!
_class
loc:@v/dense/kernel*
_output_shapes

:@*
validate_shape(*
T0
�
save_3/Assign_26Assignv/dense/kernel/Adamsave_3/RestoreV2:26*
_output_shapes

:@*
use_locking(*!
_class
loc:@v/dense/kernel*
validate_shape(*
T0
�
save_3/Assign_27Assignv/dense/kernel/Adam_1save_3/RestoreV2:27*
T0*
_output_shapes

:@*
validate_shape(*!
_class
loc:@v/dense/kernel*
use_locking(
�
save_3/Assign_28Assignv/dense_1/biassave_3/RestoreV2:28*
use_locking(*!
_class
loc:@v/dense_1/bias*
_output_shapes
:@*
validate_shape(*
T0
�
save_3/Assign_29Assignv/dense_1/bias/Adamsave_3/RestoreV2:29*!
_class
loc:@v/dense_1/bias*
use_locking(*
T0*
_output_shapes
:@*
validate_shape(
�
save_3/Assign_30Assignv/dense_1/bias/Adam_1save_3/RestoreV2:30*
use_locking(*
T0*
validate_shape(*
_output_shapes
:@*!
_class
loc:@v/dense_1/bias
�
save_3/Assign_31Assignv/dense_1/kernelsave_3/RestoreV2:31*
T0*
validate_shape(*
use_locking(*
_output_shapes

:@@*#
_class
loc:@v/dense_1/kernel
�
save_3/Assign_32Assignv/dense_1/kernel/Adamsave_3/RestoreV2:32*
_output_shapes

:@@*
T0*#
_class
loc:@v/dense_1/kernel*
validate_shape(*
use_locking(
�
save_3/Assign_33Assignv/dense_1/kernel/Adam_1save_3/RestoreV2:33*#
_class
loc:@v/dense_1/kernel*
validate_shape(*
_output_shapes

:@@*
T0*
use_locking(
�
save_3/Assign_34Assignv/dense_2/biassave_3/RestoreV2:34*
_output_shapes
:*
use_locking(*!
_class
loc:@v/dense_2/bias*
T0*
validate_shape(
�
save_3/Assign_35Assignv/dense_2/bias/Adamsave_3/RestoreV2:35*!
_class
loc:@v/dense_2/bias*
_output_shapes
:*
validate_shape(*
T0*
use_locking(
�
save_3/Assign_36Assignv/dense_2/bias/Adam_1save_3/RestoreV2:36*
use_locking(*
validate_shape(*!
_class
loc:@v/dense_2/bias*
_output_shapes
:*
T0
�
save_3/Assign_37Assignv/dense_2/kernelsave_3/RestoreV2:37*
validate_shape(*
T0*
_output_shapes

:@*#
_class
loc:@v/dense_2/kernel*
use_locking(
�
save_3/Assign_38Assignv/dense_2/kernel/Adamsave_3/RestoreV2:38*
T0*
_output_shapes

:@*
use_locking(*#
_class
loc:@v/dense_2/kernel*
validate_shape(
�
save_3/Assign_39Assignv/dense_2/kernel/Adam_1save_3/RestoreV2:39*
T0*#
_class
loc:@v/dense_2/kernel*
use_locking(*
_output_shapes

:@*
validate_shape(
�
save_3/restore_shardNoOp^save_3/Assign^save_3/Assign_1^save_3/Assign_10^save_3/Assign_11^save_3/Assign_12^save_3/Assign_13^save_3/Assign_14^save_3/Assign_15^save_3/Assign_16^save_3/Assign_17^save_3/Assign_18^save_3/Assign_19^save_3/Assign_2^save_3/Assign_20^save_3/Assign_21^save_3/Assign_22^save_3/Assign_23^save_3/Assign_24^save_3/Assign_25^save_3/Assign_26^save_3/Assign_27^save_3/Assign_28^save_3/Assign_29^save_3/Assign_3^save_3/Assign_30^save_3/Assign_31^save_3/Assign_32^save_3/Assign_33^save_3/Assign_34^save_3/Assign_35^save_3/Assign_36^save_3/Assign_37^save_3/Assign_38^save_3/Assign_39^save_3/Assign_4^save_3/Assign_5^save_3/Assign_6^save_3/Assign_7^save_3/Assign_8^save_3/Assign_9
1
save_3/restore_allNoOp^save_3/restore_shard
[
save_4/filename/inputConst*
dtype0*
valueB Bmodel*
_output_shapes
: 
r
save_4/filenamePlaceholderWithDefaultsave_4/filename/input*
_output_shapes
: *
shape: *
dtype0
i
save_4/ConstPlaceholderWithDefaultsave_4/filename*
_output_shapes
: *
shape: *
dtype0
�
save_4/StringJoin/inputs_1Const*
dtype0*
_output_shapes
: *<
value3B1 B+_temp_a12be3d471d24d3cb373deb239c53621/part
{
save_4/StringJoin
StringJoinsave_4/Constsave_4/StringJoin/inputs_1*
_output_shapes
: *
	separator *
N
S
save_4/num_shardsConst*
_output_shapes
: *
dtype0*
value	B :
^
save_4/ShardedFilename/shardConst*
dtype0*
value	B : *
_output_shapes
: 
�
save_4/ShardedFilenameShardedFilenamesave_4/StringJoinsave_4/ShardedFilename/shardsave_4/num_shards*
_output_shapes
: 
�
save_4/SaveV2/tensor_namesConst*�
value�B�(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:(
�
save_4/SaveV2/shape_and_slicesConst*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:(
�
save_4/SaveV2SaveV2save_4/ShardedFilenamesave_4/SaveV2/tensor_namessave_4/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1v/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1v/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1v/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1v/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1v/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1v/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1*6
dtypes,
*2(
�
save_4/control_dependencyIdentitysave_4/ShardedFilename^save_4/SaveV2*
_output_shapes
: *)
_class
loc:@save_4/ShardedFilename*
T0
�
-save_4/MergeV2Checkpoints/checkpoint_prefixesPacksave_4/ShardedFilename^save_4/control_dependency*

axis *
N*
T0*
_output_shapes
:
�
save_4/MergeV2CheckpointsMergeV2Checkpoints-save_4/MergeV2Checkpoints/checkpoint_prefixessave_4/Const*
delete_old_dirs(
�
save_4/IdentityIdentitysave_4/Const^save_4/MergeV2Checkpoints^save_4/control_dependency*
T0*
_output_shapes
: 
�
save_4/RestoreV2/tensor_namesConst*�
value�B�(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:(
�
!save_4/RestoreV2/shape_and_slicesConst*
dtype0*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:(
�
save_4/RestoreV2	RestoreV2save_4/Constsave_4/RestoreV2/tensor_names!save_4/RestoreV2/shape_and_slices*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::*6
dtypes,
*2(
�
save_4/AssignAssignbeta1_powersave_4/RestoreV2*
_output_shapes
: * 
_class
loc:@pi/dense/bias*
use_locking(*
validate_shape(*
T0
�
save_4/Assign_1Assignbeta1_power_1save_4/RestoreV2:1*
_output_shapes
: *
use_locking(*
_class
loc:@v/dense/bias*
validate_shape(*
T0
�
save_4/Assign_2Assignbeta2_powersave_4/RestoreV2:2* 
_class
loc:@pi/dense/bias*
_output_shapes
: *
T0*
use_locking(*
validate_shape(
�
save_4/Assign_3Assignbeta2_power_1save_4/RestoreV2:3*
use_locking(*
_output_shapes
: *
T0*
validate_shape(*
_class
loc:@v/dense/bias
�
save_4/Assign_4Assignpi/dense/biassave_4/RestoreV2:4*
T0*
validate_shape(*
_output_shapes
:@* 
_class
loc:@pi/dense/bias*
use_locking(
�
save_4/Assign_5Assignpi/dense/bias/Adamsave_4/RestoreV2:5*
_output_shapes
:@*
validate_shape(*
use_locking(*
T0* 
_class
loc:@pi/dense/bias
�
save_4/Assign_6Assignpi/dense/bias/Adam_1save_4/RestoreV2:6*
T0*
use_locking(*
_output_shapes
:@* 
_class
loc:@pi/dense/bias*
validate_shape(
�
save_4/Assign_7Assignpi/dense/kernelsave_4/RestoreV2:7*
T0*
use_locking(*
_output_shapes

:@*
validate_shape(*"
_class
loc:@pi/dense/kernel
�
save_4/Assign_8Assignpi/dense/kernel/Adamsave_4/RestoreV2:8*"
_class
loc:@pi/dense/kernel*
T0*
use_locking(*
validate_shape(*
_output_shapes

:@
�
save_4/Assign_9Assignpi/dense/kernel/Adam_1save_4/RestoreV2:9*
use_locking(*
validate_shape(*
T0*"
_class
loc:@pi/dense/kernel*
_output_shapes

:@
�
save_4/Assign_10Assignpi/dense_1/biassave_4/RestoreV2:10*
T0*"
_class
loc:@pi/dense_1/bias*
use_locking(*
_output_shapes
:@*
validate_shape(
�
save_4/Assign_11Assignpi/dense_1/bias/Adamsave_4/RestoreV2:11*
validate_shape(*
_output_shapes
:@*"
_class
loc:@pi/dense_1/bias*
T0*
use_locking(
�
save_4/Assign_12Assignpi/dense_1/bias/Adam_1save_4/RestoreV2:12*
_output_shapes
:@*
validate_shape(*"
_class
loc:@pi/dense_1/bias*
T0*
use_locking(
�
save_4/Assign_13Assignpi/dense_1/kernelsave_4/RestoreV2:13*$
_class
loc:@pi/dense_1/kernel*
T0*
_output_shapes

:@@*
validate_shape(*
use_locking(
�
save_4/Assign_14Assignpi/dense_1/kernel/Adamsave_4/RestoreV2:14*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel*
_output_shapes

:@@*
validate_shape(
�
save_4/Assign_15Assignpi/dense_1/kernel/Adam_1save_4/RestoreV2:15*
validate_shape(*
use_locking(*$
_class
loc:@pi/dense_1/kernel*
_output_shapes

:@@*
T0
�
save_4/Assign_16Assignpi/dense_2/biassave_4/RestoreV2:16*
_output_shapes
:*
validate_shape(*
use_locking(*"
_class
loc:@pi/dense_2/bias*
T0
�
save_4/Assign_17Assignpi/dense_2/bias/Adamsave_4/RestoreV2:17*
_output_shapes
:*
validate_shape(*"
_class
loc:@pi/dense_2/bias*
use_locking(*
T0
�
save_4/Assign_18Assignpi/dense_2/bias/Adam_1save_4/RestoreV2:18*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
use_locking(*
T0*
_output_shapes
:
�
save_4/Assign_19Assignpi/dense_2/kernelsave_4/RestoreV2:19*
validate_shape(*
T0*$
_class
loc:@pi/dense_2/kernel*
use_locking(*
_output_shapes

:@
�
save_4/Assign_20Assignpi/dense_2/kernel/Adamsave_4/RestoreV2:20*
validate_shape(*$
_class
loc:@pi/dense_2/kernel*
T0*
use_locking(*
_output_shapes

:@
�
save_4/Assign_21Assignpi/dense_2/kernel/Adam_1save_4/RestoreV2:21*
T0*
use_locking(*
validate_shape(*$
_class
loc:@pi/dense_2/kernel*
_output_shapes

:@
�
save_4/Assign_22Assignv/dense/biassave_4/RestoreV2:22*
T0*
_output_shapes
:@*
validate_shape(*
use_locking(*
_class
loc:@v/dense/bias
�
save_4/Assign_23Assignv/dense/bias/Adamsave_4/RestoreV2:23*
validate_shape(*
T0*
_class
loc:@v/dense/bias*
_output_shapes
:@*
use_locking(
�
save_4/Assign_24Assignv/dense/bias/Adam_1save_4/RestoreV2:24*
T0*
_class
loc:@v/dense/bias*
_output_shapes
:@*
use_locking(*
validate_shape(
�
save_4/Assign_25Assignv/dense/kernelsave_4/RestoreV2:25*
use_locking(*
_output_shapes

:@*
validate_shape(*!
_class
loc:@v/dense/kernel*
T0
�
save_4/Assign_26Assignv/dense/kernel/Adamsave_4/RestoreV2:26*
_output_shapes

:@*
T0*!
_class
loc:@v/dense/kernel*
validate_shape(*
use_locking(
�
save_4/Assign_27Assignv/dense/kernel/Adam_1save_4/RestoreV2:27*!
_class
loc:@v/dense/kernel*
validate_shape(*
use_locking(*
T0*
_output_shapes

:@
�
save_4/Assign_28Assignv/dense_1/biassave_4/RestoreV2:28*
validate_shape(*
_output_shapes
:@*!
_class
loc:@v/dense_1/bias*
T0*
use_locking(
�
save_4/Assign_29Assignv/dense_1/bias/Adamsave_4/RestoreV2:29*
_output_shapes
:@*
validate_shape(*
T0*
use_locking(*!
_class
loc:@v/dense_1/bias
�
save_4/Assign_30Assignv/dense_1/bias/Adam_1save_4/RestoreV2:30*
T0*
_output_shapes
:@*
validate_shape(*!
_class
loc:@v/dense_1/bias*
use_locking(
�
save_4/Assign_31Assignv/dense_1/kernelsave_4/RestoreV2:31*
use_locking(*
validate_shape(*#
_class
loc:@v/dense_1/kernel*
T0*
_output_shapes

:@@
�
save_4/Assign_32Assignv/dense_1/kernel/Adamsave_4/RestoreV2:32*
validate_shape(*
use_locking(*#
_class
loc:@v/dense_1/kernel*
_output_shapes

:@@*
T0
�
save_4/Assign_33Assignv/dense_1/kernel/Adam_1save_4/RestoreV2:33*
use_locking(*
T0*
_output_shapes

:@@*#
_class
loc:@v/dense_1/kernel*
validate_shape(
�
save_4/Assign_34Assignv/dense_2/biassave_4/RestoreV2:34*
use_locking(*
validate_shape(*
_output_shapes
:*!
_class
loc:@v/dense_2/bias*
T0
�
save_4/Assign_35Assignv/dense_2/bias/Adamsave_4/RestoreV2:35*
use_locking(*!
_class
loc:@v/dense_2/bias*
_output_shapes
:*
T0*
validate_shape(
�
save_4/Assign_36Assignv/dense_2/bias/Adam_1save_4/RestoreV2:36*!
_class
loc:@v/dense_2/bias*
_output_shapes
:*
use_locking(*
validate_shape(*
T0
�
save_4/Assign_37Assignv/dense_2/kernelsave_4/RestoreV2:37*#
_class
loc:@v/dense_2/kernel*
_output_shapes

:@*
use_locking(*
validate_shape(*
T0
�
save_4/Assign_38Assignv/dense_2/kernel/Adamsave_4/RestoreV2:38*
use_locking(*#
_class
loc:@v/dense_2/kernel*
_output_shapes

:@*
validate_shape(*
T0
�
save_4/Assign_39Assignv/dense_2/kernel/Adam_1save_4/RestoreV2:39*
validate_shape(*
use_locking(*
_output_shapes

:@*
T0*#
_class
loc:@v/dense_2/kernel
�
save_4/restore_shardNoOp^save_4/Assign^save_4/Assign_1^save_4/Assign_10^save_4/Assign_11^save_4/Assign_12^save_4/Assign_13^save_4/Assign_14^save_4/Assign_15^save_4/Assign_16^save_4/Assign_17^save_4/Assign_18^save_4/Assign_19^save_4/Assign_2^save_4/Assign_20^save_4/Assign_21^save_4/Assign_22^save_4/Assign_23^save_4/Assign_24^save_4/Assign_25^save_4/Assign_26^save_4/Assign_27^save_4/Assign_28^save_4/Assign_29^save_4/Assign_3^save_4/Assign_30^save_4/Assign_31^save_4/Assign_32^save_4/Assign_33^save_4/Assign_34^save_4/Assign_35^save_4/Assign_36^save_4/Assign_37^save_4/Assign_38^save_4/Assign_39^save_4/Assign_4^save_4/Assign_5^save_4/Assign_6^save_4/Assign_7^save_4/Assign_8^save_4/Assign_9
1
save_4/restore_allNoOp^save_4/restore_shard
[
save_5/filename/inputConst*
dtype0*
valueB Bmodel*
_output_shapes
: 
r
save_5/filenamePlaceholderWithDefaultsave_5/filename/input*
_output_shapes
: *
shape: *
dtype0
i
save_5/ConstPlaceholderWithDefaultsave_5/filename*
shape: *
_output_shapes
: *
dtype0
�
save_5/StringJoin/inputs_1Const*
dtype0*
_output_shapes
: *<
value3B1 B+_temp_888c539e8df24d5e8aae36070ab04a1d/part
{
save_5/StringJoin
StringJoinsave_5/Constsave_5/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
S
save_5/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
^
save_5/ShardedFilename/shardConst*
_output_shapes
: *
value	B : *
dtype0
�
save_5/ShardedFilenameShardedFilenamesave_5/StringJoinsave_5/ShardedFilename/shardsave_5/num_shards*
_output_shapes
: 
�
save_5/SaveV2/tensor_namesConst*�
value�B�(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:(
�
save_5/SaveV2/shape_and_slicesConst*
dtype0*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:(
�
save_5/SaveV2SaveV2save_5/ShardedFilenamesave_5/SaveV2/tensor_namessave_5/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1v/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1v/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1v/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1v/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1v/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1v/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1*6
dtypes,
*2(
�
save_5/control_dependencyIdentitysave_5/ShardedFilename^save_5/SaveV2*)
_class
loc:@save_5/ShardedFilename*
T0*
_output_shapes
: 
�
-save_5/MergeV2Checkpoints/checkpoint_prefixesPacksave_5/ShardedFilename^save_5/control_dependency*

axis *
N*
T0*
_output_shapes
:
�
save_5/MergeV2CheckpointsMergeV2Checkpoints-save_5/MergeV2Checkpoints/checkpoint_prefixessave_5/Const*
delete_old_dirs(
�
save_5/IdentityIdentitysave_5/Const^save_5/MergeV2Checkpoints^save_5/control_dependency*
_output_shapes
: *
T0
�
save_5/RestoreV2/tensor_namesConst*�
value�B�(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:(
�
!save_5/RestoreV2/shape_and_slicesConst*
dtype0*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:(
�
save_5/RestoreV2	RestoreV2save_5/Constsave_5/RestoreV2/tensor_names!save_5/RestoreV2/shape_and_slices*6
dtypes,
*2(*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::
�
save_5/AssignAssignbeta1_powersave_5/RestoreV2*
validate_shape(*
T0*
use_locking(*
_output_shapes
: * 
_class
loc:@pi/dense/bias
�
save_5/Assign_1Assignbeta1_power_1save_5/RestoreV2:1*
_output_shapes
: *
_class
loc:@v/dense/bias*
use_locking(*
T0*
validate_shape(
�
save_5/Assign_2Assignbeta2_powersave_5/RestoreV2:2*
use_locking(*
_output_shapes
: *
T0*
validate_shape(* 
_class
loc:@pi/dense/bias
�
save_5/Assign_3Assignbeta2_power_1save_5/RestoreV2:3*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@v/dense/bias*
validate_shape(
�
save_5/Assign_4Assignpi/dense/biassave_5/RestoreV2:4* 
_class
loc:@pi/dense/bias*
T0*
_output_shapes
:@*
validate_shape(*
use_locking(
�
save_5/Assign_5Assignpi/dense/bias/Adamsave_5/RestoreV2:5*
_output_shapes
:@*
use_locking(*
T0*
validate_shape(* 
_class
loc:@pi/dense/bias
�
save_5/Assign_6Assignpi/dense/bias/Adam_1save_5/RestoreV2:6*
T0*
validate_shape(*
use_locking(*
_output_shapes
:@* 
_class
loc:@pi/dense/bias
�
save_5/Assign_7Assignpi/dense/kernelsave_5/RestoreV2:7*
use_locking(*"
_class
loc:@pi/dense/kernel*
T0*
validate_shape(*
_output_shapes

:@
�
save_5/Assign_8Assignpi/dense/kernel/Adamsave_5/RestoreV2:8*
use_locking(*
T0*
_output_shapes

:@*"
_class
loc:@pi/dense/kernel*
validate_shape(
�
save_5/Assign_9Assignpi/dense/kernel/Adam_1save_5/RestoreV2:9*"
_class
loc:@pi/dense/kernel*
_output_shapes

:@*
use_locking(*
T0*
validate_shape(
�
save_5/Assign_10Assignpi/dense_1/biassave_5/RestoreV2:10*
use_locking(*
validate_shape(*"
_class
loc:@pi/dense_1/bias*
_output_shapes
:@*
T0
�
save_5/Assign_11Assignpi/dense_1/bias/Adamsave_5/RestoreV2:11*"
_class
loc:@pi/dense_1/bias*
use_locking(*
_output_shapes
:@*
T0*
validate_shape(
�
save_5/Assign_12Assignpi/dense_1/bias/Adam_1save_5/RestoreV2:12*
T0*
_output_shapes
:@*
validate_shape(*"
_class
loc:@pi/dense_1/bias*
use_locking(
�
save_5/Assign_13Assignpi/dense_1/kernelsave_5/RestoreV2:13*
_output_shapes

:@@*$
_class
loc:@pi/dense_1/kernel*
validate_shape(*
use_locking(*
T0
�
save_5/Assign_14Assignpi/dense_1/kernel/Adamsave_5/RestoreV2:14*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel*
_output_shapes

:@@*
validate_shape(
�
save_5/Assign_15Assignpi/dense_1/kernel/Adam_1save_5/RestoreV2:15*
validate_shape(*$
_class
loc:@pi/dense_1/kernel*
use_locking(*
_output_shapes

:@@*
T0
�
save_5/Assign_16Assignpi/dense_2/biassave_5/RestoreV2:16*
use_locking(*
validate_shape(*"
_class
loc:@pi/dense_2/bias*
T0*
_output_shapes
:
�
save_5/Assign_17Assignpi/dense_2/bias/Adamsave_5/RestoreV2:17*
T0*"
_class
loc:@pi/dense_2/bias*
use_locking(*
validate_shape(*
_output_shapes
:
�
save_5/Assign_18Assignpi/dense_2/bias/Adam_1save_5/RestoreV2:18*
T0*
_output_shapes
:*
use_locking(*
validate_shape(*"
_class
loc:@pi/dense_2/bias
�
save_5/Assign_19Assignpi/dense_2/kernelsave_5/RestoreV2:19*
validate_shape(*$
_class
loc:@pi/dense_2/kernel*
_output_shapes

:@*
T0*
use_locking(
�
save_5/Assign_20Assignpi/dense_2/kernel/Adamsave_5/RestoreV2:20*
_output_shapes

:@*
use_locking(*
validate_shape(*
T0*$
_class
loc:@pi/dense_2/kernel
�
save_5/Assign_21Assignpi/dense_2/kernel/Adam_1save_5/RestoreV2:21*
_output_shapes

:@*
use_locking(*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
T0
�
save_5/Assign_22Assignv/dense/biassave_5/RestoreV2:22*
T0*
validate_shape(*
_output_shapes
:@*
use_locking(*
_class
loc:@v/dense/bias
�
save_5/Assign_23Assignv/dense/bias/Adamsave_5/RestoreV2:23*
validate_shape(*
_class
loc:@v/dense/bias*
use_locking(*
T0*
_output_shapes
:@
�
save_5/Assign_24Assignv/dense/bias/Adam_1save_5/RestoreV2:24*
validate_shape(*
_class
loc:@v/dense/bias*
use_locking(*
T0*
_output_shapes
:@
�
save_5/Assign_25Assignv/dense/kernelsave_5/RestoreV2:25*
validate_shape(*!
_class
loc:@v/dense/kernel*
_output_shapes

:@*
use_locking(*
T0
�
save_5/Assign_26Assignv/dense/kernel/Adamsave_5/RestoreV2:26*
T0*
validate_shape(*
_output_shapes

:@*
use_locking(*!
_class
loc:@v/dense/kernel
�
save_5/Assign_27Assignv/dense/kernel/Adam_1save_5/RestoreV2:27*
_output_shapes

:@*
validate_shape(*
T0*
use_locking(*!
_class
loc:@v/dense/kernel
�
save_5/Assign_28Assignv/dense_1/biassave_5/RestoreV2:28*
_output_shapes
:@*
validate_shape(*
T0*
use_locking(*!
_class
loc:@v/dense_1/bias
�
save_5/Assign_29Assignv/dense_1/bias/Adamsave_5/RestoreV2:29*
T0*
_output_shapes
:@*
validate_shape(*
use_locking(*!
_class
loc:@v/dense_1/bias
�
save_5/Assign_30Assignv/dense_1/bias/Adam_1save_5/RestoreV2:30*
use_locking(*!
_class
loc:@v/dense_1/bias*
_output_shapes
:@*
T0*
validate_shape(
�
save_5/Assign_31Assignv/dense_1/kernelsave_5/RestoreV2:31*#
_class
loc:@v/dense_1/kernel*
_output_shapes

:@@*
validate_shape(*
use_locking(*
T0
�
save_5/Assign_32Assignv/dense_1/kernel/Adamsave_5/RestoreV2:32*
use_locking(*
T0*
validate_shape(*#
_class
loc:@v/dense_1/kernel*
_output_shapes

:@@
�
save_5/Assign_33Assignv/dense_1/kernel/Adam_1save_5/RestoreV2:33*
_output_shapes

:@@*#
_class
loc:@v/dense_1/kernel*
use_locking(*
validate_shape(*
T0
�
save_5/Assign_34Assignv/dense_2/biassave_5/RestoreV2:34*
_output_shapes
:*
validate_shape(*!
_class
loc:@v/dense_2/bias*
T0*
use_locking(
�
save_5/Assign_35Assignv/dense_2/bias/Adamsave_5/RestoreV2:35*
_output_shapes
:*
use_locking(*
T0*
validate_shape(*!
_class
loc:@v/dense_2/bias
�
save_5/Assign_36Assignv/dense_2/bias/Adam_1save_5/RestoreV2:36*
T0*
use_locking(*!
_class
loc:@v/dense_2/bias*
_output_shapes
:*
validate_shape(
�
save_5/Assign_37Assignv/dense_2/kernelsave_5/RestoreV2:37*
_output_shapes

:@*
use_locking(*
validate_shape(*#
_class
loc:@v/dense_2/kernel*
T0
�
save_5/Assign_38Assignv/dense_2/kernel/Adamsave_5/RestoreV2:38*
use_locking(*
T0*
validate_shape(*#
_class
loc:@v/dense_2/kernel*
_output_shapes

:@
�
save_5/Assign_39Assignv/dense_2/kernel/Adam_1save_5/RestoreV2:39*
validate_shape(*
_output_shapes

:@*
T0*
use_locking(*#
_class
loc:@v/dense_2/kernel
�
save_5/restore_shardNoOp^save_5/Assign^save_5/Assign_1^save_5/Assign_10^save_5/Assign_11^save_5/Assign_12^save_5/Assign_13^save_5/Assign_14^save_5/Assign_15^save_5/Assign_16^save_5/Assign_17^save_5/Assign_18^save_5/Assign_19^save_5/Assign_2^save_5/Assign_20^save_5/Assign_21^save_5/Assign_22^save_5/Assign_23^save_5/Assign_24^save_5/Assign_25^save_5/Assign_26^save_5/Assign_27^save_5/Assign_28^save_5/Assign_29^save_5/Assign_3^save_5/Assign_30^save_5/Assign_31^save_5/Assign_32^save_5/Assign_33^save_5/Assign_34^save_5/Assign_35^save_5/Assign_36^save_5/Assign_37^save_5/Assign_38^save_5/Assign_39^save_5/Assign_4^save_5/Assign_5^save_5/Assign_6^save_5/Assign_7^save_5/Assign_8^save_5/Assign_9
1
save_5/restore_allNoOp^save_5/restore_shard
[
save_6/filename/inputConst*
dtype0*
_output_shapes
: *
valueB Bmodel
r
save_6/filenamePlaceholderWithDefaultsave_6/filename/input*
dtype0*
shape: *
_output_shapes
: 
i
save_6/ConstPlaceholderWithDefaultsave_6/filename*
dtype0*
shape: *
_output_shapes
: 
�
save_6/StringJoin/inputs_1Const*
dtype0*
_output_shapes
: *<
value3B1 B+_temp_53975697e16944658704b2c2a44d0221/part
{
save_6/StringJoin
StringJoinsave_6/Constsave_6/StringJoin/inputs_1*
	separator *
N*
_output_shapes
: 
S
save_6/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
^
save_6/ShardedFilename/shardConst*
_output_shapes
: *
value	B : *
dtype0
�
save_6/ShardedFilenameShardedFilenamesave_6/StringJoinsave_6/ShardedFilename/shardsave_6/num_shards*
_output_shapes
: 
�
save_6/SaveV2/tensor_namesConst*�
value�B�(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
_output_shapes
:(*
dtype0
�
save_6/SaveV2/shape_and_slicesConst*
_output_shapes
:(*
dtype0*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
�
save_6/SaveV2SaveV2save_6/ShardedFilenamesave_6/SaveV2/tensor_namessave_6/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1v/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1v/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1v/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1v/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1v/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1v/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1*6
dtypes,
*2(
�
save_6/control_dependencyIdentitysave_6/ShardedFilename^save_6/SaveV2*
_output_shapes
: *)
_class
loc:@save_6/ShardedFilename*
T0
�
-save_6/MergeV2Checkpoints/checkpoint_prefixesPacksave_6/ShardedFilename^save_6/control_dependency*
_output_shapes
:*
N*

axis *
T0
�
save_6/MergeV2CheckpointsMergeV2Checkpoints-save_6/MergeV2Checkpoints/checkpoint_prefixessave_6/Const*
delete_old_dirs(
�
save_6/IdentityIdentitysave_6/Const^save_6/MergeV2Checkpoints^save_6/control_dependency*
_output_shapes
: *
T0
�
save_6/RestoreV2/tensor_namesConst*�
value�B�(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:(
�
!save_6/RestoreV2/shape_and_slicesConst*
_output_shapes
:(*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
�
save_6/RestoreV2	RestoreV2save_6/Constsave_6/RestoreV2/tensor_names!save_6/RestoreV2/shape_and_slices*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::*6
dtypes,
*2(
�
save_6/AssignAssignbeta1_powersave_6/RestoreV2*
validate_shape(* 
_class
loc:@pi/dense/bias*
T0*
use_locking(*
_output_shapes
: 
�
save_6/Assign_1Assignbeta1_power_1save_6/RestoreV2:1*
_output_shapes
: *
_class
loc:@v/dense/bias*
validate_shape(*
T0*
use_locking(
�
save_6/Assign_2Assignbeta2_powersave_6/RestoreV2:2* 
_class
loc:@pi/dense/bias*
_output_shapes
: *
validate_shape(*
use_locking(*
T0
�
save_6/Assign_3Assignbeta2_power_1save_6/RestoreV2:3*
_output_shapes
: *
_class
loc:@v/dense/bias*
validate_shape(*
T0*
use_locking(
�
save_6/Assign_4Assignpi/dense/biassave_6/RestoreV2:4*
use_locking(*
_output_shapes
:@*
validate_shape(* 
_class
loc:@pi/dense/bias*
T0
�
save_6/Assign_5Assignpi/dense/bias/Adamsave_6/RestoreV2:5*
use_locking(*
validate_shape(*
T0*
_output_shapes
:@* 
_class
loc:@pi/dense/bias
�
save_6/Assign_6Assignpi/dense/bias/Adam_1save_6/RestoreV2:6* 
_class
loc:@pi/dense/bias*
T0*
_output_shapes
:@*
use_locking(*
validate_shape(
�
save_6/Assign_7Assignpi/dense/kernelsave_6/RestoreV2:7*
use_locking(*"
_class
loc:@pi/dense/kernel*
_output_shapes

:@*
T0*
validate_shape(
�
save_6/Assign_8Assignpi/dense/kernel/Adamsave_6/RestoreV2:8*
T0*"
_class
loc:@pi/dense/kernel*
use_locking(*
_output_shapes

:@*
validate_shape(
�
save_6/Assign_9Assignpi/dense/kernel/Adam_1save_6/RestoreV2:9*
_output_shapes

:@*
T0*
use_locking(*"
_class
loc:@pi/dense/kernel*
validate_shape(
�
save_6/Assign_10Assignpi/dense_1/biassave_6/RestoreV2:10*"
_class
loc:@pi/dense_1/bias*
use_locking(*
_output_shapes
:@*
validate_shape(*
T0
�
save_6/Assign_11Assignpi/dense_1/bias/Adamsave_6/RestoreV2:11*
_output_shapes
:@*
use_locking(*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
T0
�
save_6/Assign_12Assignpi/dense_1/bias/Adam_1save_6/RestoreV2:12*
validate_shape(*
_output_shapes
:@*
T0*"
_class
loc:@pi/dense_1/bias*
use_locking(
�
save_6/Assign_13Assignpi/dense_1/kernelsave_6/RestoreV2:13*$
_class
loc:@pi/dense_1/kernel*
use_locking(*
_output_shapes

:@@*
validate_shape(*
T0
�
save_6/Assign_14Assignpi/dense_1/kernel/Adamsave_6/RestoreV2:14*$
_class
loc:@pi/dense_1/kernel*
T0*
validate_shape(*
use_locking(*
_output_shapes

:@@
�
save_6/Assign_15Assignpi/dense_1/kernel/Adam_1save_6/RestoreV2:15*$
_class
loc:@pi/dense_1/kernel*
T0*
validate_shape(*
_output_shapes

:@@*
use_locking(
�
save_6/Assign_16Assignpi/dense_2/biassave_6/RestoreV2:16*
validate_shape(*
T0*
use_locking(*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:
�
save_6/Assign_17Assignpi/dense_2/bias/Adamsave_6/RestoreV2:17*
validate_shape(*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:*
T0*
use_locking(
�
save_6/Assign_18Assignpi/dense_2/bias/Adam_1save_6/RestoreV2:18*
T0*
_output_shapes
:*
use_locking(*
validate_shape(*"
_class
loc:@pi/dense_2/bias
�
save_6/Assign_19Assignpi/dense_2/kernelsave_6/RestoreV2:19*$
_class
loc:@pi/dense_2/kernel*
T0*
use_locking(*
_output_shapes

:@*
validate_shape(
�
save_6/Assign_20Assignpi/dense_2/kernel/Adamsave_6/RestoreV2:20*
T0*
use_locking(*
validate_shape(*
_output_shapes

:@*$
_class
loc:@pi/dense_2/kernel
�
save_6/Assign_21Assignpi/dense_2/kernel/Adam_1save_6/RestoreV2:21*
use_locking(*
T0*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@
�
save_6/Assign_22Assignv/dense/biassave_6/RestoreV2:22*
T0*
use_locking(*
_output_shapes
:@*
validate_shape(*
_class
loc:@v/dense/bias
�
save_6/Assign_23Assignv/dense/bias/Adamsave_6/RestoreV2:23*
use_locking(*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
:@
�
save_6/Assign_24Assignv/dense/bias/Adam_1save_6/RestoreV2:24*
_class
loc:@v/dense/bias*
validate_shape(*
use_locking(*
_output_shapes
:@*
T0
�
save_6/Assign_25Assignv/dense/kernelsave_6/RestoreV2:25*
validate_shape(*
T0*
_output_shapes

:@*!
_class
loc:@v/dense/kernel*
use_locking(
�
save_6/Assign_26Assignv/dense/kernel/Adamsave_6/RestoreV2:26*
T0*!
_class
loc:@v/dense/kernel*
use_locking(*
_output_shapes

:@*
validate_shape(
�
save_6/Assign_27Assignv/dense/kernel/Adam_1save_6/RestoreV2:27*
_output_shapes

:@*
T0*
validate_shape(*!
_class
loc:@v/dense/kernel*
use_locking(
�
save_6/Assign_28Assignv/dense_1/biassave_6/RestoreV2:28*
_output_shapes
:@*!
_class
loc:@v/dense_1/bias*
T0*
validate_shape(*
use_locking(
�
save_6/Assign_29Assignv/dense_1/bias/Adamsave_6/RestoreV2:29*
_output_shapes
:@*
validate_shape(*
T0*!
_class
loc:@v/dense_1/bias*
use_locking(
�
save_6/Assign_30Assignv/dense_1/bias/Adam_1save_6/RestoreV2:30*!
_class
loc:@v/dense_1/bias*
_output_shapes
:@*
validate_shape(*
T0*
use_locking(
�
save_6/Assign_31Assignv/dense_1/kernelsave_6/RestoreV2:31*
validate_shape(*
T0*#
_class
loc:@v/dense_1/kernel*
_output_shapes

:@@*
use_locking(
�
save_6/Assign_32Assignv/dense_1/kernel/Adamsave_6/RestoreV2:32*
use_locking(*#
_class
loc:@v/dense_1/kernel*
T0*
validate_shape(*
_output_shapes

:@@
�
save_6/Assign_33Assignv/dense_1/kernel/Adam_1save_6/RestoreV2:33*#
_class
loc:@v/dense_1/kernel*
_output_shapes

:@@*
use_locking(*
T0*
validate_shape(
�
save_6/Assign_34Assignv/dense_2/biassave_6/RestoreV2:34*
T0*
_output_shapes
:*
validate_shape(*!
_class
loc:@v/dense_2/bias*
use_locking(
�
save_6/Assign_35Assignv/dense_2/bias/Adamsave_6/RestoreV2:35*
use_locking(*!
_class
loc:@v/dense_2/bias*
T0*
_output_shapes
:*
validate_shape(
�
save_6/Assign_36Assignv/dense_2/bias/Adam_1save_6/RestoreV2:36*!
_class
loc:@v/dense_2/bias*
validate_shape(*
T0*
_output_shapes
:*
use_locking(
�
save_6/Assign_37Assignv/dense_2/kernelsave_6/RestoreV2:37*
use_locking(*
T0*
_output_shapes

:@*#
_class
loc:@v/dense_2/kernel*
validate_shape(
�
save_6/Assign_38Assignv/dense_2/kernel/Adamsave_6/RestoreV2:38*
validate_shape(*
T0*
_output_shapes

:@*#
_class
loc:@v/dense_2/kernel*
use_locking(
�
save_6/Assign_39Assignv/dense_2/kernel/Adam_1save_6/RestoreV2:39*#
_class
loc:@v/dense_2/kernel*
use_locking(*
validate_shape(*
_output_shapes

:@*
T0
�
save_6/restore_shardNoOp^save_6/Assign^save_6/Assign_1^save_6/Assign_10^save_6/Assign_11^save_6/Assign_12^save_6/Assign_13^save_6/Assign_14^save_6/Assign_15^save_6/Assign_16^save_6/Assign_17^save_6/Assign_18^save_6/Assign_19^save_6/Assign_2^save_6/Assign_20^save_6/Assign_21^save_6/Assign_22^save_6/Assign_23^save_6/Assign_24^save_6/Assign_25^save_6/Assign_26^save_6/Assign_27^save_6/Assign_28^save_6/Assign_29^save_6/Assign_3^save_6/Assign_30^save_6/Assign_31^save_6/Assign_32^save_6/Assign_33^save_6/Assign_34^save_6/Assign_35^save_6/Assign_36^save_6/Assign_37^save_6/Assign_38^save_6/Assign_39^save_6/Assign_4^save_6/Assign_5^save_6/Assign_6^save_6/Assign_7^save_6/Assign_8^save_6/Assign_9
1
save_6/restore_allNoOp^save_6/restore_shard
[
save_7/filename/inputConst*
dtype0*
valueB Bmodel*
_output_shapes
: 
r
save_7/filenamePlaceholderWithDefaultsave_7/filename/input*
dtype0*
shape: *
_output_shapes
: 
i
save_7/ConstPlaceholderWithDefaultsave_7/filename*
dtype0*
_output_shapes
: *
shape: 
�
save_7/StringJoin/inputs_1Const*<
value3B1 B+_temp_4b520df7616a4192b7c0f3d4b93ee3f6/part*
dtype0*
_output_shapes
: 
{
save_7/StringJoin
StringJoinsave_7/Constsave_7/StringJoin/inputs_1*
	separator *
N*
_output_shapes
: 
S
save_7/num_shardsConst*
_output_shapes
: *
dtype0*
value	B :
^
save_7/ShardedFilename/shardConst*
dtype0*
_output_shapes
: *
value	B : 
�
save_7/ShardedFilenameShardedFilenamesave_7/StringJoinsave_7/ShardedFilename/shardsave_7/num_shards*
_output_shapes
: 
�
save_7/SaveV2/tensor_namesConst*
_output_shapes
:(*
dtype0*�
value�B�(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1
�
save_7/SaveV2/shape_and_slicesConst*
dtype0*
_output_shapes
:(*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
�
save_7/SaveV2SaveV2save_7/ShardedFilenamesave_7/SaveV2/tensor_namessave_7/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1v/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1v/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1v/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1v/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1v/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1v/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1*6
dtypes,
*2(
�
save_7/control_dependencyIdentitysave_7/ShardedFilename^save_7/SaveV2*
_output_shapes
: *
T0*)
_class
loc:@save_7/ShardedFilename
�
-save_7/MergeV2Checkpoints/checkpoint_prefixesPacksave_7/ShardedFilename^save_7/control_dependency*

axis *
_output_shapes
:*
N*
T0
�
save_7/MergeV2CheckpointsMergeV2Checkpoints-save_7/MergeV2Checkpoints/checkpoint_prefixessave_7/Const*
delete_old_dirs(
�
save_7/IdentityIdentitysave_7/Const^save_7/MergeV2Checkpoints^save_7/control_dependency*
T0*
_output_shapes
: 
�
save_7/RestoreV2/tensor_namesConst*�
value�B�(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
_output_shapes
:(*
dtype0
�
!save_7/RestoreV2/shape_and_slicesConst*
_output_shapes
:(*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
�
save_7/RestoreV2	RestoreV2save_7/Constsave_7/RestoreV2/tensor_names!save_7/RestoreV2/shape_and_slices*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::*6
dtypes,
*2(
�
save_7/AssignAssignbeta1_powersave_7/RestoreV2* 
_class
loc:@pi/dense/bias*
_output_shapes
: *
T0*
use_locking(*
validate_shape(
�
save_7/Assign_1Assignbeta1_power_1save_7/RestoreV2:1*
use_locking(*
validate_shape(*
_output_shapes
: *
_class
loc:@v/dense/bias*
T0
�
save_7/Assign_2Assignbeta2_powersave_7/RestoreV2:2*
use_locking(*
validate_shape(*
_output_shapes
: * 
_class
loc:@pi/dense/bias*
T0
�
save_7/Assign_3Assignbeta2_power_1save_7/RestoreV2:3*
use_locking(*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
: *
T0
�
save_7/Assign_4Assignpi/dense/biassave_7/RestoreV2:4*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
use_locking(*
_output_shapes
:@
�
save_7/Assign_5Assignpi/dense/bias/Adamsave_7/RestoreV2:5*
T0*
_output_shapes
:@*
use_locking(*
validate_shape(* 
_class
loc:@pi/dense/bias
�
save_7/Assign_6Assignpi/dense/bias/Adam_1save_7/RestoreV2:6*
validate_shape(*
_output_shapes
:@*
T0*
use_locking(* 
_class
loc:@pi/dense/bias
�
save_7/Assign_7Assignpi/dense/kernelsave_7/RestoreV2:7*
use_locking(*
_output_shapes

:@*
validate_shape(*"
_class
loc:@pi/dense/kernel*
T0
�
save_7/Assign_8Assignpi/dense/kernel/Adamsave_7/RestoreV2:8*
use_locking(*
validate_shape(*
T0*"
_class
loc:@pi/dense/kernel*
_output_shapes

:@
�
save_7/Assign_9Assignpi/dense/kernel/Adam_1save_7/RestoreV2:9*
validate_shape(*
_output_shapes

:@*
use_locking(*"
_class
loc:@pi/dense/kernel*
T0
�
save_7/Assign_10Assignpi/dense_1/biassave_7/RestoreV2:10*
T0*
_output_shapes
:@*"
_class
loc:@pi/dense_1/bias*
use_locking(*
validate_shape(
�
save_7/Assign_11Assignpi/dense_1/bias/Adamsave_7/RestoreV2:11*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
T0*
_output_shapes
:@*
use_locking(
�
save_7/Assign_12Assignpi/dense_1/bias/Adam_1save_7/RestoreV2:12*
_output_shapes
:@*
validate_shape(*
T0*
use_locking(*"
_class
loc:@pi/dense_1/bias
�
save_7/Assign_13Assignpi/dense_1/kernelsave_7/RestoreV2:13*
_output_shapes

:@@*
validate_shape(*$
_class
loc:@pi/dense_1/kernel*
T0*
use_locking(
�
save_7/Assign_14Assignpi/dense_1/kernel/Adamsave_7/RestoreV2:14*
T0*
validate_shape(*
use_locking(*
_output_shapes

:@@*$
_class
loc:@pi/dense_1/kernel
�
save_7/Assign_15Assignpi/dense_1/kernel/Adam_1save_7/RestoreV2:15*
T0*$
_class
loc:@pi/dense_1/kernel*
_output_shapes

:@@*
use_locking(*
validate_shape(
�
save_7/Assign_16Assignpi/dense_2/biassave_7/RestoreV2:16*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:*
use_locking(*
T0*
validate_shape(
�
save_7/Assign_17Assignpi/dense_2/bias/Adamsave_7/RestoreV2:17*"
_class
loc:@pi/dense_2/bias*
T0*
_output_shapes
:*
validate_shape(*
use_locking(
�
save_7/Assign_18Assignpi/dense_2/bias/Adam_1save_7/RestoreV2:18*
T0*
validate_shape(*
_output_shapes
:*"
_class
loc:@pi/dense_2/bias*
use_locking(
�
save_7/Assign_19Assignpi/dense_2/kernelsave_7/RestoreV2:19*
validate_shape(*$
_class
loc:@pi/dense_2/kernel*
_output_shapes

:@*
use_locking(*
T0
�
save_7/Assign_20Assignpi/dense_2/kernel/Adamsave_7/RestoreV2:20*
_output_shapes

:@*$
_class
loc:@pi/dense_2/kernel*
T0*
use_locking(*
validate_shape(
�
save_7/Assign_21Assignpi/dense_2/kernel/Adam_1save_7/RestoreV2:21*
T0*$
_class
loc:@pi/dense_2/kernel*
use_locking(*
_output_shapes

:@*
validate_shape(
�
save_7/Assign_22Assignv/dense/biassave_7/RestoreV2:22*
T0*
_output_shapes
:@*
_class
loc:@v/dense/bias*
validate_shape(*
use_locking(
�
save_7/Assign_23Assignv/dense/bias/Adamsave_7/RestoreV2:23*
use_locking(*
_class
loc:@v/dense/bias*
_output_shapes
:@*
validate_shape(*
T0
�
save_7/Assign_24Assignv/dense/bias/Adam_1save_7/RestoreV2:24*
_class
loc:@v/dense/bias*
validate_shape(*
use_locking(*
_output_shapes
:@*
T0
�
save_7/Assign_25Assignv/dense/kernelsave_7/RestoreV2:25*
use_locking(*
_output_shapes

:@*
validate_shape(*!
_class
loc:@v/dense/kernel*
T0
�
save_7/Assign_26Assignv/dense/kernel/Adamsave_7/RestoreV2:26*
_output_shapes

:@*
use_locking(*
T0*!
_class
loc:@v/dense/kernel*
validate_shape(
�
save_7/Assign_27Assignv/dense/kernel/Adam_1save_7/RestoreV2:27*
_output_shapes

:@*
use_locking(*!
_class
loc:@v/dense/kernel*
validate_shape(*
T0
�
save_7/Assign_28Assignv/dense_1/biassave_7/RestoreV2:28*!
_class
loc:@v/dense_1/bias*
T0*
validate_shape(*
_output_shapes
:@*
use_locking(
�
save_7/Assign_29Assignv/dense_1/bias/Adamsave_7/RestoreV2:29*!
_class
loc:@v/dense_1/bias*
validate_shape(*
use_locking(*
T0*
_output_shapes
:@
�
save_7/Assign_30Assignv/dense_1/bias/Adam_1save_7/RestoreV2:30*
validate_shape(*
use_locking(*
T0*!
_class
loc:@v/dense_1/bias*
_output_shapes
:@
�
save_7/Assign_31Assignv/dense_1/kernelsave_7/RestoreV2:31*
_output_shapes

:@@*
T0*
validate_shape(*#
_class
loc:@v/dense_1/kernel*
use_locking(
�
save_7/Assign_32Assignv/dense_1/kernel/Adamsave_7/RestoreV2:32*
T0*#
_class
loc:@v/dense_1/kernel*
_output_shapes

:@@*
validate_shape(*
use_locking(
�
save_7/Assign_33Assignv/dense_1/kernel/Adam_1save_7/RestoreV2:33*
T0*
validate_shape(*#
_class
loc:@v/dense_1/kernel*
_output_shapes

:@@*
use_locking(
�
save_7/Assign_34Assignv/dense_2/biassave_7/RestoreV2:34*!
_class
loc:@v/dense_2/bias*
_output_shapes
:*
validate_shape(*
T0*
use_locking(
�
save_7/Assign_35Assignv/dense_2/bias/Adamsave_7/RestoreV2:35*!
_class
loc:@v/dense_2/bias*
_output_shapes
:*
T0*
use_locking(*
validate_shape(
�
save_7/Assign_36Assignv/dense_2/bias/Adam_1save_7/RestoreV2:36*
use_locking(*
T0*!
_class
loc:@v/dense_2/bias*
_output_shapes
:*
validate_shape(
�
save_7/Assign_37Assignv/dense_2/kernelsave_7/RestoreV2:37*
_output_shapes

:@*
validate_shape(*
T0*
use_locking(*#
_class
loc:@v/dense_2/kernel
�
save_7/Assign_38Assignv/dense_2/kernel/Adamsave_7/RestoreV2:38*
_output_shapes

:@*
validate_shape(*#
_class
loc:@v/dense_2/kernel*
use_locking(*
T0
�
save_7/Assign_39Assignv/dense_2/kernel/Adam_1save_7/RestoreV2:39*
use_locking(*#
_class
loc:@v/dense_2/kernel*
_output_shapes

:@*
T0*
validate_shape(
�
save_7/restore_shardNoOp^save_7/Assign^save_7/Assign_1^save_7/Assign_10^save_7/Assign_11^save_7/Assign_12^save_7/Assign_13^save_7/Assign_14^save_7/Assign_15^save_7/Assign_16^save_7/Assign_17^save_7/Assign_18^save_7/Assign_19^save_7/Assign_2^save_7/Assign_20^save_7/Assign_21^save_7/Assign_22^save_7/Assign_23^save_7/Assign_24^save_7/Assign_25^save_7/Assign_26^save_7/Assign_27^save_7/Assign_28^save_7/Assign_29^save_7/Assign_3^save_7/Assign_30^save_7/Assign_31^save_7/Assign_32^save_7/Assign_33^save_7/Assign_34^save_7/Assign_35^save_7/Assign_36^save_7/Assign_37^save_7/Assign_38^save_7/Assign_39^save_7/Assign_4^save_7/Assign_5^save_7/Assign_6^save_7/Assign_7^save_7/Assign_8^save_7/Assign_9
1
save_7/restore_allNoOp^save_7/restore_shard
[
save_8/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
r
save_8/filenamePlaceholderWithDefaultsave_8/filename/input*
_output_shapes
: *
shape: *
dtype0
i
save_8/ConstPlaceholderWithDefaultsave_8/filename*
dtype0*
shape: *
_output_shapes
: 
�
save_8/StringJoin/inputs_1Const*
dtype0*<
value3B1 B+_temp_52f8548a54d946dea5c37a2432a1780e/part*
_output_shapes
: 
{
save_8/StringJoin
StringJoinsave_8/Constsave_8/StringJoin/inputs_1*
	separator *
_output_shapes
: *
N
S
save_8/num_shardsConst*
_output_shapes
: *
value	B :*
dtype0
^
save_8/ShardedFilename/shardConst*
dtype0*
_output_shapes
: *
value	B : 
�
save_8/ShardedFilenameShardedFilenamesave_8/StringJoinsave_8/ShardedFilename/shardsave_8/num_shards*
_output_shapes
: 
�
save_8/SaveV2/tensor_namesConst*
_output_shapes
:(*
dtype0*�
value�B�(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1
�
save_8/SaveV2/shape_and_slicesConst*
_output_shapes
:(*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
�
save_8/SaveV2SaveV2save_8/ShardedFilenamesave_8/SaveV2/tensor_namessave_8/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1v/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1v/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1v/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1v/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1v/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1v/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1*6
dtypes,
*2(
�
save_8/control_dependencyIdentitysave_8/ShardedFilename^save_8/SaveV2*)
_class
loc:@save_8/ShardedFilename*
_output_shapes
: *
T0
�
-save_8/MergeV2Checkpoints/checkpoint_prefixesPacksave_8/ShardedFilename^save_8/control_dependency*
_output_shapes
:*
N*

axis *
T0
�
save_8/MergeV2CheckpointsMergeV2Checkpoints-save_8/MergeV2Checkpoints/checkpoint_prefixessave_8/Const*
delete_old_dirs(
�
save_8/IdentityIdentitysave_8/Const^save_8/MergeV2Checkpoints^save_8/control_dependency*
_output_shapes
: *
T0
�
save_8/RestoreV2/tensor_namesConst*
_output_shapes
:(*
dtype0*�
value�B�(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1
�
!save_8/RestoreV2/shape_and_slicesConst*
dtype0*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:(
�
save_8/RestoreV2	RestoreV2save_8/Constsave_8/RestoreV2/tensor_names!save_8/RestoreV2/shape_and_slices*6
dtypes,
*2(*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::
�
save_8/AssignAssignbeta1_powersave_8/RestoreV2*
_output_shapes
: *
validate_shape(*
T0* 
_class
loc:@pi/dense/bias*
use_locking(
�
save_8/Assign_1Assignbeta1_power_1save_8/RestoreV2:1*
validate_shape(*
T0*
use_locking(*
_class
loc:@v/dense/bias*
_output_shapes
: 
�
save_8/Assign_2Assignbeta2_powersave_8/RestoreV2:2*
validate_shape(*
use_locking(*
_output_shapes
: *
T0* 
_class
loc:@pi/dense/bias
�
save_8/Assign_3Assignbeta2_power_1save_8/RestoreV2:3*
_output_shapes
: *
use_locking(*
validate_shape(*
_class
loc:@v/dense/bias*
T0
�
save_8/Assign_4Assignpi/dense/biassave_8/RestoreV2:4*
T0*
validate_shape(*
_output_shapes
:@* 
_class
loc:@pi/dense/bias*
use_locking(
�
save_8/Assign_5Assignpi/dense/bias/Adamsave_8/RestoreV2:5*
T0*
_output_shapes
:@*
validate_shape(* 
_class
loc:@pi/dense/bias*
use_locking(
�
save_8/Assign_6Assignpi/dense/bias/Adam_1save_8/RestoreV2:6*
T0*
_output_shapes
:@*
validate_shape(*
use_locking(* 
_class
loc:@pi/dense/bias
�
save_8/Assign_7Assignpi/dense/kernelsave_8/RestoreV2:7*
validate_shape(*
T0*
_output_shapes

:@*"
_class
loc:@pi/dense/kernel*
use_locking(
�
save_8/Assign_8Assignpi/dense/kernel/Adamsave_8/RestoreV2:8*
_output_shapes

:@*"
_class
loc:@pi/dense/kernel*
validate_shape(*
use_locking(*
T0
�
save_8/Assign_9Assignpi/dense/kernel/Adam_1save_8/RestoreV2:9*"
_class
loc:@pi/dense/kernel*
use_locking(*
T0*
validate_shape(*
_output_shapes

:@
�
save_8/Assign_10Assignpi/dense_1/biassave_8/RestoreV2:10*
T0*
validate_shape(*"
_class
loc:@pi/dense_1/bias*
use_locking(*
_output_shapes
:@
�
save_8/Assign_11Assignpi/dense_1/bias/Adamsave_8/RestoreV2:11*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
T0*
use_locking(*
_output_shapes
:@
�
save_8/Assign_12Assignpi/dense_1/bias/Adam_1save_8/RestoreV2:12*
T0*
_output_shapes
:@*
use_locking(*
validate_shape(*"
_class
loc:@pi/dense_1/bias
�
save_8/Assign_13Assignpi/dense_1/kernelsave_8/RestoreV2:13*
use_locking(*$
_class
loc:@pi/dense_1/kernel*
T0*
validate_shape(*
_output_shapes

:@@
�
save_8/Assign_14Assignpi/dense_1/kernel/Adamsave_8/RestoreV2:14*$
_class
loc:@pi/dense_1/kernel*
_output_shapes

:@@*
T0*
validate_shape(*
use_locking(
�
save_8/Assign_15Assignpi/dense_1/kernel/Adam_1save_8/RestoreV2:15*$
_class
loc:@pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@*
T0*
use_locking(
�
save_8/Assign_16Assignpi/dense_2/biassave_8/RestoreV2:16*
validate_shape(*"
_class
loc:@pi/dense_2/bias*
T0*
use_locking(*
_output_shapes
:
�
save_8/Assign_17Assignpi/dense_2/bias/Adamsave_8/RestoreV2:17*"
_class
loc:@pi/dense_2/bias*
use_locking(*
_output_shapes
:*
validate_shape(*
T0
�
save_8/Assign_18Assignpi/dense_2/bias/Adam_1save_8/RestoreV2:18*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
T0*
use_locking(*
_output_shapes
:
�
save_8/Assign_19Assignpi/dense_2/kernelsave_8/RestoreV2:19*
_output_shapes

:@*
use_locking(*
validate_shape(*
T0*$
_class
loc:@pi/dense_2/kernel
�
save_8/Assign_20Assignpi/dense_2/kernel/Adamsave_8/RestoreV2:20*
use_locking(*
_output_shapes

:@*
validate_shape(*$
_class
loc:@pi/dense_2/kernel*
T0
�
save_8/Assign_21Assignpi/dense_2/kernel/Adam_1save_8/RestoreV2:21*
use_locking(*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
T0*
_output_shapes

:@
�
save_8/Assign_22Assignv/dense/biassave_8/RestoreV2:22*
use_locking(*
_output_shapes
:@*
validate_shape(*
_class
loc:@v/dense/bias*
T0
�
save_8/Assign_23Assignv/dense/bias/Adamsave_8/RestoreV2:23*
_output_shapes
:@*
_class
loc:@v/dense/bias*
T0*
validate_shape(*
use_locking(
�
save_8/Assign_24Assignv/dense/bias/Adam_1save_8/RestoreV2:24*
validate_shape(*
T0*
_class
loc:@v/dense/bias*
_output_shapes
:@*
use_locking(
�
save_8/Assign_25Assignv/dense/kernelsave_8/RestoreV2:25*
_output_shapes

:@*
validate_shape(*
use_locking(*!
_class
loc:@v/dense/kernel*
T0
�
save_8/Assign_26Assignv/dense/kernel/Adamsave_8/RestoreV2:26*
validate_shape(*
T0*
_output_shapes

:@*
use_locking(*!
_class
loc:@v/dense/kernel
�
save_8/Assign_27Assignv/dense/kernel/Adam_1save_8/RestoreV2:27*
use_locking(*!
_class
loc:@v/dense/kernel*
_output_shapes

:@*
validate_shape(*
T0
�
save_8/Assign_28Assignv/dense_1/biassave_8/RestoreV2:28*
T0*
use_locking(*
validate_shape(*!
_class
loc:@v/dense_1/bias*
_output_shapes
:@
�
save_8/Assign_29Assignv/dense_1/bias/Adamsave_8/RestoreV2:29*!
_class
loc:@v/dense_1/bias*
_output_shapes
:@*
use_locking(*
validate_shape(*
T0
�
save_8/Assign_30Assignv/dense_1/bias/Adam_1save_8/RestoreV2:30*
validate_shape(*!
_class
loc:@v/dense_1/bias*
_output_shapes
:@*
use_locking(*
T0
�
save_8/Assign_31Assignv/dense_1/kernelsave_8/RestoreV2:31*
T0*
use_locking(*
validate_shape(*#
_class
loc:@v/dense_1/kernel*
_output_shapes

:@@
�
save_8/Assign_32Assignv/dense_1/kernel/Adamsave_8/RestoreV2:32*#
_class
loc:@v/dense_1/kernel*
validate_shape(*
T0*
use_locking(*
_output_shapes

:@@
�
save_8/Assign_33Assignv/dense_1/kernel/Adam_1save_8/RestoreV2:33*#
_class
loc:@v/dense_1/kernel*
T0*
validate_shape(*
use_locking(*
_output_shapes

:@@
�
save_8/Assign_34Assignv/dense_2/biassave_8/RestoreV2:34*
_output_shapes
:*!
_class
loc:@v/dense_2/bias*
use_locking(*
validate_shape(*
T0
�
save_8/Assign_35Assignv/dense_2/bias/Adamsave_8/RestoreV2:35*
T0*
use_locking(*
validate_shape(*
_output_shapes
:*!
_class
loc:@v/dense_2/bias
�
save_8/Assign_36Assignv/dense_2/bias/Adam_1save_8/RestoreV2:36*
validate_shape(*
T0*
use_locking(*!
_class
loc:@v/dense_2/bias*
_output_shapes
:
�
save_8/Assign_37Assignv/dense_2/kernelsave_8/RestoreV2:37*#
_class
loc:@v/dense_2/kernel*
_output_shapes

:@*
validate_shape(*
use_locking(*
T0
�
save_8/Assign_38Assignv/dense_2/kernel/Adamsave_8/RestoreV2:38*
_output_shapes

:@*#
_class
loc:@v/dense_2/kernel*
T0*
validate_shape(*
use_locking(
�
save_8/Assign_39Assignv/dense_2/kernel/Adam_1save_8/RestoreV2:39*
use_locking(*
_output_shapes

:@*
validate_shape(*
T0*#
_class
loc:@v/dense_2/kernel
�
save_8/restore_shardNoOp^save_8/Assign^save_8/Assign_1^save_8/Assign_10^save_8/Assign_11^save_8/Assign_12^save_8/Assign_13^save_8/Assign_14^save_8/Assign_15^save_8/Assign_16^save_8/Assign_17^save_8/Assign_18^save_8/Assign_19^save_8/Assign_2^save_8/Assign_20^save_8/Assign_21^save_8/Assign_22^save_8/Assign_23^save_8/Assign_24^save_8/Assign_25^save_8/Assign_26^save_8/Assign_27^save_8/Assign_28^save_8/Assign_29^save_8/Assign_3^save_8/Assign_30^save_8/Assign_31^save_8/Assign_32^save_8/Assign_33^save_8/Assign_34^save_8/Assign_35^save_8/Assign_36^save_8/Assign_37^save_8/Assign_38^save_8/Assign_39^save_8/Assign_4^save_8/Assign_5^save_8/Assign_6^save_8/Assign_7^save_8/Assign_8^save_8/Assign_9
1
save_8/restore_allNoOp^save_8/restore_shard
[
save_9/filename/inputConst*
_output_shapes
: *
valueB Bmodel*
dtype0
r
save_9/filenamePlaceholderWithDefaultsave_9/filename/input*
dtype0*
_output_shapes
: *
shape: 
i
save_9/ConstPlaceholderWithDefaultsave_9/filename*
dtype0*
shape: *
_output_shapes
: 
�
save_9/StringJoin/inputs_1Const*
_output_shapes
: *<
value3B1 B+_temp_1941796bc7674f0b9024592315f272d7/part*
dtype0
{
save_9/StringJoin
StringJoinsave_9/Constsave_9/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
S
save_9/num_shardsConst*
dtype0*
value	B :*
_output_shapes
: 
^
save_9/ShardedFilename/shardConst*
dtype0*
_output_shapes
: *
value	B : 
�
save_9/ShardedFilenameShardedFilenamesave_9/StringJoinsave_9/ShardedFilename/shardsave_9/num_shards*
_output_shapes
: 
�
save_9/SaveV2/tensor_namesConst*
_output_shapes
:(*�
value�B�(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
dtype0
�
save_9/SaveV2/shape_and_slicesConst*
_output_shapes
:(*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
�
save_9/SaveV2SaveV2save_9/ShardedFilenamesave_9/SaveV2/tensor_namessave_9/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1v/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1v/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1v/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1v/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1v/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1v/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1*6
dtypes,
*2(
�
save_9/control_dependencyIdentitysave_9/ShardedFilename^save_9/SaveV2*
T0*
_output_shapes
: *)
_class
loc:@save_9/ShardedFilename
�
-save_9/MergeV2Checkpoints/checkpoint_prefixesPacksave_9/ShardedFilename^save_9/control_dependency*
T0*
_output_shapes
:*

axis *
N
�
save_9/MergeV2CheckpointsMergeV2Checkpoints-save_9/MergeV2Checkpoints/checkpoint_prefixessave_9/Const*
delete_old_dirs(
�
save_9/IdentityIdentitysave_9/Const^save_9/MergeV2Checkpoints^save_9/control_dependency*
_output_shapes
: *
T0
�
save_9/RestoreV2/tensor_namesConst*
dtype0*
_output_shapes
:(*�
value�B�(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1
�
!save_9/RestoreV2/shape_and_slicesConst*
dtype0*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:(
�
save_9/RestoreV2	RestoreV2save_9/Constsave_9/RestoreV2/tensor_names!save_9/RestoreV2/shape_and_slices*6
dtypes,
*2(*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::
�
save_9/AssignAssignbeta1_powersave_9/RestoreV2*
T0* 
_class
loc:@pi/dense/bias*
use_locking(*
validate_shape(*
_output_shapes
: 
�
save_9/Assign_1Assignbeta1_power_1save_9/RestoreV2:1*
T0*
_output_shapes
: *
validate_shape(*
use_locking(*
_class
loc:@v/dense/bias
�
save_9/Assign_2Assignbeta2_powersave_9/RestoreV2:2*
use_locking(*
_output_shapes
: *
T0*
validate_shape(* 
_class
loc:@pi/dense/bias
�
save_9/Assign_3Assignbeta2_power_1save_9/RestoreV2:3*
validate_shape(*
_output_shapes
: *
T0*
_class
loc:@v/dense/bias*
use_locking(
�
save_9/Assign_4Assignpi/dense/biassave_9/RestoreV2:4*
validate_shape(*
use_locking(*
_output_shapes
:@* 
_class
loc:@pi/dense/bias*
T0
�
save_9/Assign_5Assignpi/dense/bias/Adamsave_9/RestoreV2:5* 
_class
loc:@pi/dense/bias*
_output_shapes
:@*
use_locking(*
validate_shape(*
T0
�
save_9/Assign_6Assignpi/dense/bias/Adam_1save_9/RestoreV2:6*
validate_shape(*
_output_shapes
:@* 
_class
loc:@pi/dense/bias*
T0*
use_locking(
�
save_9/Assign_7Assignpi/dense/kernelsave_9/RestoreV2:7*
use_locking(*
_output_shapes

:@*
validate_shape(*"
_class
loc:@pi/dense/kernel*
T0
�
save_9/Assign_8Assignpi/dense/kernel/Adamsave_9/RestoreV2:8*
validate_shape(*"
_class
loc:@pi/dense/kernel*
T0*
use_locking(*
_output_shapes

:@
�
save_9/Assign_9Assignpi/dense/kernel/Adam_1save_9/RestoreV2:9*
validate_shape(*
T0*"
_class
loc:@pi/dense/kernel*
use_locking(*
_output_shapes

:@
�
save_9/Assign_10Assignpi/dense_1/biassave_9/RestoreV2:10*
_output_shapes
:@*
T0*
use_locking(*
validate_shape(*"
_class
loc:@pi/dense_1/bias
�
save_9/Assign_11Assignpi/dense_1/bias/Adamsave_9/RestoreV2:11*
use_locking(*
validate_shape(*
T0*"
_class
loc:@pi/dense_1/bias*
_output_shapes
:@
�
save_9/Assign_12Assignpi/dense_1/bias/Adam_1save_9/RestoreV2:12*"
_class
loc:@pi/dense_1/bias*
_output_shapes
:@*
validate_shape(*
use_locking(*
T0
�
save_9/Assign_13Assignpi/dense_1/kernelsave_9/RestoreV2:13*$
_class
loc:@pi/dense_1/kernel*
validate_shape(*
T0*
_output_shapes

:@@*
use_locking(
�
save_9/Assign_14Assignpi/dense_1/kernel/Adamsave_9/RestoreV2:14*
use_locking(*$
_class
loc:@pi/dense_1/kernel*
T0*
_output_shapes

:@@*
validate_shape(
�
save_9/Assign_15Assignpi/dense_1/kernel/Adam_1save_9/RestoreV2:15*$
_class
loc:@pi/dense_1/kernel*
use_locking(*
_output_shapes

:@@*
validate_shape(*
T0
�
save_9/Assign_16Assignpi/dense_2/biassave_9/RestoreV2:16*
T0*
_output_shapes
:*
validate_shape(*"
_class
loc:@pi/dense_2/bias*
use_locking(
�
save_9/Assign_17Assignpi/dense_2/bias/Adamsave_9/RestoreV2:17*
_output_shapes
:*
T0*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
use_locking(
�
save_9/Assign_18Assignpi/dense_2/bias/Adam_1save_9/RestoreV2:18*
_output_shapes
:*
T0*
use_locking(*"
_class
loc:@pi/dense_2/bias*
validate_shape(
�
save_9/Assign_19Assignpi/dense_2/kernelsave_9/RestoreV2:19*$
_class
loc:@pi/dense_2/kernel*
use_locking(*
T0*
validate_shape(*
_output_shapes

:@
�
save_9/Assign_20Assignpi/dense_2/kernel/Adamsave_9/RestoreV2:20*
validate_shape(*
T0*$
_class
loc:@pi/dense_2/kernel*
_output_shapes

:@*
use_locking(
�
save_9/Assign_21Assignpi/dense_2/kernel/Adam_1save_9/RestoreV2:21*
use_locking(*$
_class
loc:@pi/dense_2/kernel*
_output_shapes

:@*
T0*
validate_shape(
�
save_9/Assign_22Assignv/dense/biassave_9/RestoreV2:22*
_output_shapes
:@*
use_locking(*
T0*
_class
loc:@v/dense/bias*
validate_shape(
�
save_9/Assign_23Assignv/dense/bias/Adamsave_9/RestoreV2:23*
_output_shapes
:@*
use_locking(*
_class
loc:@v/dense/bias*
T0*
validate_shape(
�
save_9/Assign_24Assignv/dense/bias/Adam_1save_9/RestoreV2:24*
_output_shapes
:@*
T0*
use_locking(*
validate_shape(*
_class
loc:@v/dense/bias
�
save_9/Assign_25Assignv/dense/kernelsave_9/RestoreV2:25*
_output_shapes

:@*
validate_shape(*!
_class
loc:@v/dense/kernel*
use_locking(*
T0
�
save_9/Assign_26Assignv/dense/kernel/Adamsave_9/RestoreV2:26*
use_locking(*
validate_shape(*
_output_shapes

:@*!
_class
loc:@v/dense/kernel*
T0
�
save_9/Assign_27Assignv/dense/kernel/Adam_1save_9/RestoreV2:27*
_output_shapes

:@*!
_class
loc:@v/dense/kernel*
use_locking(*
T0*
validate_shape(
�
save_9/Assign_28Assignv/dense_1/biassave_9/RestoreV2:28*
_output_shapes
:@*!
_class
loc:@v/dense_1/bias*
use_locking(*
validate_shape(*
T0
�
save_9/Assign_29Assignv/dense_1/bias/Adamsave_9/RestoreV2:29*!
_class
loc:@v/dense_1/bias*
validate_shape(*
T0*
_output_shapes
:@*
use_locking(
�
save_9/Assign_30Assignv/dense_1/bias/Adam_1save_9/RestoreV2:30*
_output_shapes
:@*
use_locking(*
validate_shape(*!
_class
loc:@v/dense_1/bias*
T0
�
save_9/Assign_31Assignv/dense_1/kernelsave_9/RestoreV2:31*#
_class
loc:@v/dense_1/kernel*
validate_shape(*
use_locking(*
T0*
_output_shapes

:@@
�
save_9/Assign_32Assignv/dense_1/kernel/Adamsave_9/RestoreV2:32*
_output_shapes

:@@*
T0*
use_locking(*
validate_shape(*#
_class
loc:@v/dense_1/kernel
�
save_9/Assign_33Assignv/dense_1/kernel/Adam_1save_9/RestoreV2:33*
use_locking(*
_output_shapes

:@@*#
_class
loc:@v/dense_1/kernel*
validate_shape(*
T0
�
save_9/Assign_34Assignv/dense_2/biassave_9/RestoreV2:34*
use_locking(*
_output_shapes
:*
T0*
validate_shape(*!
_class
loc:@v/dense_2/bias
�
save_9/Assign_35Assignv/dense_2/bias/Adamsave_9/RestoreV2:35*
T0*
_output_shapes
:*
validate_shape(*!
_class
loc:@v/dense_2/bias*
use_locking(
�
save_9/Assign_36Assignv/dense_2/bias/Adam_1save_9/RestoreV2:36*
_output_shapes
:*
validate_shape(*!
_class
loc:@v/dense_2/bias*
T0*
use_locking(
�
save_9/Assign_37Assignv/dense_2/kernelsave_9/RestoreV2:37*
validate_shape(*#
_class
loc:@v/dense_2/kernel*
T0*
_output_shapes

:@*
use_locking(
�
save_9/Assign_38Assignv/dense_2/kernel/Adamsave_9/RestoreV2:38*
validate_shape(*
T0*#
_class
loc:@v/dense_2/kernel*
use_locking(*
_output_shapes

:@
�
save_9/Assign_39Assignv/dense_2/kernel/Adam_1save_9/RestoreV2:39*
_output_shapes

:@*
T0*
validate_shape(*#
_class
loc:@v/dense_2/kernel*
use_locking(
�
save_9/restore_shardNoOp^save_9/Assign^save_9/Assign_1^save_9/Assign_10^save_9/Assign_11^save_9/Assign_12^save_9/Assign_13^save_9/Assign_14^save_9/Assign_15^save_9/Assign_16^save_9/Assign_17^save_9/Assign_18^save_9/Assign_19^save_9/Assign_2^save_9/Assign_20^save_9/Assign_21^save_9/Assign_22^save_9/Assign_23^save_9/Assign_24^save_9/Assign_25^save_9/Assign_26^save_9/Assign_27^save_9/Assign_28^save_9/Assign_29^save_9/Assign_3^save_9/Assign_30^save_9/Assign_31^save_9/Assign_32^save_9/Assign_33^save_9/Assign_34^save_9/Assign_35^save_9/Assign_36^save_9/Assign_37^save_9/Assign_38^save_9/Assign_39^save_9/Assign_4^save_9/Assign_5^save_9/Assign_6^save_9/Assign_7^save_9/Assign_8^save_9/Assign_9
1
save_9/restore_allNoOp^save_9/restore_shard
\
save_10/filename/inputConst*
_output_shapes
: *
valueB Bmodel*
dtype0
t
save_10/filenamePlaceholderWithDefaultsave_10/filename/input*
dtype0*
_output_shapes
: *
shape: 
k
save_10/ConstPlaceholderWithDefaultsave_10/filename*
shape: *
dtype0*
_output_shapes
: 
�
save_10/StringJoin/inputs_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+_temp_9351ccdab7da4f6eb819c9db32992576/part
~
save_10/StringJoin
StringJoinsave_10/Constsave_10/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
T
save_10/num_shardsConst*
dtype0*
value	B :*
_output_shapes
: 
_
save_10/ShardedFilename/shardConst*
_output_shapes
: *
value	B : *
dtype0
�
save_10/ShardedFilenameShardedFilenamesave_10/StringJoinsave_10/ShardedFilename/shardsave_10/num_shards*
_output_shapes
: 
�
save_10/SaveV2/tensor_namesConst*�
value�B�(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
_output_shapes
:(*
dtype0
�
save_10/SaveV2/shape_and_slicesConst*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:(
�
save_10/SaveV2SaveV2save_10/ShardedFilenamesave_10/SaveV2/tensor_namessave_10/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1v/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1v/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1v/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1v/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1v/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1v/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1*6
dtypes,
*2(
�
save_10/control_dependencyIdentitysave_10/ShardedFilename^save_10/SaveV2**
_class 
loc:@save_10/ShardedFilename*
T0*
_output_shapes
: 
�
.save_10/MergeV2Checkpoints/checkpoint_prefixesPacksave_10/ShardedFilename^save_10/control_dependency*
N*
T0*
_output_shapes
:*

axis 
�
save_10/MergeV2CheckpointsMergeV2Checkpoints.save_10/MergeV2Checkpoints/checkpoint_prefixessave_10/Const*
delete_old_dirs(
�
save_10/IdentityIdentitysave_10/Const^save_10/MergeV2Checkpoints^save_10/control_dependency*
T0*
_output_shapes
: 
�
save_10/RestoreV2/tensor_namesConst*�
value�B�(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
_output_shapes
:(*
dtype0
�
"save_10/RestoreV2/shape_and_slicesConst*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:(
�
save_10/RestoreV2	RestoreV2save_10/Constsave_10/RestoreV2/tensor_names"save_10/RestoreV2/shape_and_slices*6
dtypes,
*2(*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::
�
save_10/AssignAssignbeta1_powersave_10/RestoreV2*
_output_shapes
: * 
_class
loc:@pi/dense/bias*
T0*
use_locking(*
validate_shape(
�
save_10/Assign_1Assignbeta1_power_1save_10/RestoreV2:1*
_class
loc:@v/dense/bias*
use_locking(*
_output_shapes
: *
T0*
validate_shape(
�
save_10/Assign_2Assignbeta2_powersave_10/RestoreV2:2*
T0*
validate_shape(* 
_class
loc:@pi/dense/bias*
_output_shapes
: *
use_locking(
�
save_10/Assign_3Assignbeta2_power_1save_10/RestoreV2:3*
validate_shape(*
use_locking(*
_class
loc:@v/dense/bias*
T0*
_output_shapes
: 
�
save_10/Assign_4Assignpi/dense/biassave_10/RestoreV2:4*
_output_shapes
:@* 
_class
loc:@pi/dense/bias*
use_locking(*
validate_shape(*
T0
�
save_10/Assign_5Assignpi/dense/bias/Adamsave_10/RestoreV2:5*
_output_shapes
:@*
use_locking(* 
_class
loc:@pi/dense/bias*
validate_shape(*
T0
�
save_10/Assign_6Assignpi/dense/bias/Adam_1save_10/RestoreV2:6*
validate_shape(*
_output_shapes
:@*
T0* 
_class
loc:@pi/dense/bias*
use_locking(
�
save_10/Assign_7Assignpi/dense/kernelsave_10/RestoreV2:7*
_output_shapes

:@*
use_locking(*"
_class
loc:@pi/dense/kernel*
validate_shape(*
T0
�
save_10/Assign_8Assignpi/dense/kernel/Adamsave_10/RestoreV2:8*
validate_shape(*
use_locking(*
T0*
_output_shapes

:@*"
_class
loc:@pi/dense/kernel
�
save_10/Assign_9Assignpi/dense/kernel/Adam_1save_10/RestoreV2:9*
use_locking(*
_output_shapes

:@*
validate_shape(*
T0*"
_class
loc:@pi/dense/kernel
�
save_10/Assign_10Assignpi/dense_1/biassave_10/RestoreV2:10*
use_locking(*
T0*
validate_shape(*"
_class
loc:@pi/dense_1/bias*
_output_shapes
:@
�
save_10/Assign_11Assignpi/dense_1/bias/Adamsave_10/RestoreV2:11*"
_class
loc:@pi/dense_1/bias*
T0*
_output_shapes
:@*
validate_shape(*
use_locking(
�
save_10/Assign_12Assignpi/dense_1/bias/Adam_1save_10/RestoreV2:12*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0
�
save_10/Assign_13Assignpi/dense_1/kernelsave_10/RestoreV2:13*
T0*
_output_shapes

:@@*$
_class
loc:@pi/dense_1/kernel*
use_locking(*
validate_shape(
�
save_10/Assign_14Assignpi/dense_1/kernel/Adamsave_10/RestoreV2:14*
validate_shape(*$
_class
loc:@pi/dense_1/kernel*
T0*
use_locking(*
_output_shapes

:@@
�
save_10/Assign_15Assignpi/dense_1/kernel/Adam_1save_10/RestoreV2:15*
T0*
use_locking(*
_output_shapes

:@@*
validate_shape(*$
_class
loc:@pi/dense_1/kernel
�
save_10/Assign_16Assignpi/dense_2/biassave_10/RestoreV2:16*
use_locking(*
validate_shape(*
_output_shapes
:*
T0*"
_class
loc:@pi/dense_2/bias
�
save_10/Assign_17Assignpi/dense_2/bias/Adamsave_10/RestoreV2:17*
validate_shape(*
_output_shapes
:*
T0*"
_class
loc:@pi/dense_2/bias*
use_locking(
�
save_10/Assign_18Assignpi/dense_2/bias/Adam_1save_10/RestoreV2:18*
use_locking(*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:*
validate_shape(*
T0
�
save_10/Assign_19Assignpi/dense_2/kernelsave_10/RestoreV2:19*
validate_shape(*$
_class
loc:@pi/dense_2/kernel*
use_locking(*
T0*
_output_shapes

:@
�
save_10/Assign_20Assignpi/dense_2/kernel/Adamsave_10/RestoreV2:20*$
_class
loc:@pi/dense_2/kernel*
use_locking(*
T0*
validate_shape(*
_output_shapes

:@
�
save_10/Assign_21Assignpi/dense_2/kernel/Adam_1save_10/RestoreV2:21*$
_class
loc:@pi/dense_2/kernel*
_output_shapes

:@*
use_locking(*
T0*
validate_shape(
�
save_10/Assign_22Assignv/dense/biassave_10/RestoreV2:22*
T0*
validate_shape(*
_class
loc:@v/dense/bias*
_output_shapes
:@*
use_locking(
�
save_10/Assign_23Assignv/dense/bias/Adamsave_10/RestoreV2:23*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0
�
save_10/Assign_24Assignv/dense/bias/Adam_1save_10/RestoreV2:24*
_class
loc:@v/dense/bias*
T0*
_output_shapes
:@*
validate_shape(*
use_locking(
�
save_10/Assign_25Assignv/dense/kernelsave_10/RestoreV2:25*
T0*
use_locking(*
validate_shape(*
_output_shapes

:@*!
_class
loc:@v/dense/kernel
�
save_10/Assign_26Assignv/dense/kernel/Adamsave_10/RestoreV2:26*
validate_shape(*
use_locking(*!
_class
loc:@v/dense/kernel*
T0*
_output_shapes

:@
�
save_10/Assign_27Assignv/dense/kernel/Adam_1save_10/RestoreV2:27*
use_locking(*!
_class
loc:@v/dense/kernel*
_output_shapes

:@*
validate_shape(*
T0
�
save_10/Assign_28Assignv/dense_1/biassave_10/RestoreV2:28*
T0*
_output_shapes
:@*!
_class
loc:@v/dense_1/bias*
use_locking(*
validate_shape(
�
save_10/Assign_29Assignv/dense_1/bias/Adamsave_10/RestoreV2:29*
_output_shapes
:@*!
_class
loc:@v/dense_1/bias*
validate_shape(*
T0*
use_locking(
�
save_10/Assign_30Assignv/dense_1/bias/Adam_1save_10/RestoreV2:30*!
_class
loc:@v/dense_1/bias*
T0*
use_locking(*
validate_shape(*
_output_shapes
:@
�
save_10/Assign_31Assignv/dense_1/kernelsave_10/RestoreV2:31*
use_locking(*
validate_shape(*#
_class
loc:@v/dense_1/kernel*
_output_shapes

:@@*
T0
�
save_10/Assign_32Assignv/dense_1/kernel/Adamsave_10/RestoreV2:32*
_output_shapes

:@@*#
_class
loc:@v/dense_1/kernel*
validate_shape(*
T0*
use_locking(
�
save_10/Assign_33Assignv/dense_1/kernel/Adam_1save_10/RestoreV2:33*
_output_shapes

:@@*#
_class
loc:@v/dense_1/kernel*
validate_shape(*
T0*
use_locking(
�
save_10/Assign_34Assignv/dense_2/biassave_10/RestoreV2:34*
validate_shape(*
T0*
use_locking(*
_output_shapes
:*!
_class
loc:@v/dense_2/bias
�
save_10/Assign_35Assignv/dense_2/bias/Adamsave_10/RestoreV2:35*
_output_shapes
:*
use_locking(*
validate_shape(*
T0*!
_class
loc:@v/dense_2/bias
�
save_10/Assign_36Assignv/dense_2/bias/Adam_1save_10/RestoreV2:36*
T0*
_output_shapes
:*
use_locking(*!
_class
loc:@v/dense_2/bias*
validate_shape(
�
save_10/Assign_37Assignv/dense_2/kernelsave_10/RestoreV2:37*#
_class
loc:@v/dense_2/kernel*
validate_shape(*
T0*
use_locking(*
_output_shapes

:@
�
save_10/Assign_38Assignv/dense_2/kernel/Adamsave_10/RestoreV2:38*
validate_shape(*
_output_shapes

:@*
T0*
use_locking(*#
_class
loc:@v/dense_2/kernel
�
save_10/Assign_39Assignv/dense_2/kernel/Adam_1save_10/RestoreV2:39*
_output_shapes

:@*
T0*
use_locking(*#
_class
loc:@v/dense_2/kernel*
validate_shape(
�
save_10/restore_shardNoOp^save_10/Assign^save_10/Assign_1^save_10/Assign_10^save_10/Assign_11^save_10/Assign_12^save_10/Assign_13^save_10/Assign_14^save_10/Assign_15^save_10/Assign_16^save_10/Assign_17^save_10/Assign_18^save_10/Assign_19^save_10/Assign_2^save_10/Assign_20^save_10/Assign_21^save_10/Assign_22^save_10/Assign_23^save_10/Assign_24^save_10/Assign_25^save_10/Assign_26^save_10/Assign_27^save_10/Assign_28^save_10/Assign_29^save_10/Assign_3^save_10/Assign_30^save_10/Assign_31^save_10/Assign_32^save_10/Assign_33^save_10/Assign_34^save_10/Assign_35^save_10/Assign_36^save_10/Assign_37^save_10/Assign_38^save_10/Assign_39^save_10/Assign_4^save_10/Assign_5^save_10/Assign_6^save_10/Assign_7^save_10/Assign_8^save_10/Assign_9
3
save_10/restore_allNoOp^save_10/restore_shard
\
save_11/filename/inputConst*
dtype0*
valueB Bmodel*
_output_shapes
: 
t
save_11/filenamePlaceholderWithDefaultsave_11/filename/input*
dtype0*
shape: *
_output_shapes
: 
k
save_11/ConstPlaceholderWithDefaultsave_11/filename*
shape: *
_output_shapes
: *
dtype0
�
save_11/StringJoin/inputs_1Const*
dtype0*
_output_shapes
: *<
value3B1 B+_temp_1533f49fc0ce4f858d4eadb97b3cfd6f/part
~
save_11/StringJoin
StringJoinsave_11/Constsave_11/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
T
save_11/num_shardsConst*
value	B :*
_output_shapes
: *
dtype0
_
save_11/ShardedFilename/shardConst*
value	B : *
_output_shapes
: *
dtype0
�
save_11/ShardedFilenameShardedFilenamesave_11/StringJoinsave_11/ShardedFilename/shardsave_11/num_shards*
_output_shapes
: 
�
save_11/SaveV2/tensor_namesConst*�
value�B�(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
_output_shapes
:(*
dtype0
�
save_11/SaveV2/shape_and_slicesConst*
_output_shapes
:(*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
�
save_11/SaveV2SaveV2save_11/ShardedFilenamesave_11/SaveV2/tensor_namessave_11/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1v/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1v/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1v/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1v/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1v/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1v/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1*6
dtypes,
*2(
�
save_11/control_dependencyIdentitysave_11/ShardedFilename^save_11/SaveV2*
_output_shapes
: *
T0**
_class 
loc:@save_11/ShardedFilename
�
.save_11/MergeV2Checkpoints/checkpoint_prefixesPacksave_11/ShardedFilename^save_11/control_dependency*
_output_shapes
:*

axis *
T0*
N
�
save_11/MergeV2CheckpointsMergeV2Checkpoints.save_11/MergeV2Checkpoints/checkpoint_prefixessave_11/Const*
delete_old_dirs(
�
save_11/IdentityIdentitysave_11/Const^save_11/MergeV2Checkpoints^save_11/control_dependency*
_output_shapes
: *
T0
�
save_11/RestoreV2/tensor_namesConst*
_output_shapes
:(*�
value�B�(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
dtype0
�
"save_11/RestoreV2/shape_and_slicesConst*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:(
�
save_11/RestoreV2	RestoreV2save_11/Constsave_11/RestoreV2/tensor_names"save_11/RestoreV2/shape_and_slices*6
dtypes,
*2(*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::
�
save_11/AssignAssignbeta1_powersave_11/RestoreV2*
_output_shapes
: *
T0*
use_locking(*
validate_shape(* 
_class
loc:@pi/dense/bias
�
save_11/Assign_1Assignbeta1_power_1save_11/RestoreV2:1*
use_locking(*
_output_shapes
: *
validate_shape(*
T0*
_class
loc:@v/dense/bias
�
save_11/Assign_2Assignbeta2_powersave_11/RestoreV2:2*
use_locking(*
T0*
validate_shape(* 
_class
loc:@pi/dense/bias*
_output_shapes
: 
�
save_11/Assign_3Assignbeta2_power_1save_11/RestoreV2:3*
_class
loc:@v/dense/bias*
T0*
validate_shape(*
use_locking(*
_output_shapes
: 
�
save_11/Assign_4Assignpi/dense/biassave_11/RestoreV2:4*
T0*
_output_shapes
:@*
validate_shape(* 
_class
loc:@pi/dense/bias*
use_locking(
�
save_11/Assign_5Assignpi/dense/bias/Adamsave_11/RestoreV2:5*
T0*
use_locking(* 
_class
loc:@pi/dense/bias*
_output_shapes
:@*
validate_shape(
�
save_11/Assign_6Assignpi/dense/bias/Adam_1save_11/RestoreV2:6*
T0*
validate_shape(*
use_locking(*
_output_shapes
:@* 
_class
loc:@pi/dense/bias
�
save_11/Assign_7Assignpi/dense/kernelsave_11/RestoreV2:7*"
_class
loc:@pi/dense/kernel*
_output_shapes

:@*
T0*
validate_shape(*
use_locking(
�
save_11/Assign_8Assignpi/dense/kernel/Adamsave_11/RestoreV2:8*
use_locking(*
T0*
validate_shape(*
_output_shapes

:@*"
_class
loc:@pi/dense/kernel
�
save_11/Assign_9Assignpi/dense/kernel/Adam_1save_11/RestoreV2:9*
T0*"
_class
loc:@pi/dense/kernel*
use_locking(*
_output_shapes

:@*
validate_shape(
�
save_11/Assign_10Assignpi/dense_1/biassave_11/RestoreV2:10*
T0*
validate_shape(*"
_class
loc:@pi/dense_1/bias*
use_locking(*
_output_shapes
:@
�
save_11/Assign_11Assignpi/dense_1/bias/Adamsave_11/RestoreV2:11*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
T0*
_output_shapes
:@*
use_locking(
�
save_11/Assign_12Assignpi/dense_1/bias/Adam_1save_11/RestoreV2:12*"
_class
loc:@pi/dense_1/bias*
T0*
_output_shapes
:@*
use_locking(*
validate_shape(
�
save_11/Assign_13Assignpi/dense_1/kernelsave_11/RestoreV2:13*
use_locking(*
T0*
validate_shape(*
_output_shapes

:@@*$
_class
loc:@pi/dense_1/kernel
�
save_11/Assign_14Assignpi/dense_1/kernel/Adamsave_11/RestoreV2:14*
_output_shapes

:@@*
use_locking(*$
_class
loc:@pi/dense_1/kernel*
T0*
validate_shape(
�
save_11/Assign_15Assignpi/dense_1/kernel/Adam_1save_11/RestoreV2:15*
T0*
_output_shapes

:@@*$
_class
loc:@pi/dense_1/kernel*
validate_shape(*
use_locking(
�
save_11/Assign_16Assignpi/dense_2/biassave_11/RestoreV2:16*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:*
validate_shape(
�
save_11/Assign_17Assignpi/dense_2/bias/Adamsave_11/RestoreV2:17*
use_locking(*
validate_shape(*"
_class
loc:@pi/dense_2/bias*
T0*
_output_shapes
:
�
save_11/Assign_18Assignpi/dense_2/bias/Adam_1save_11/RestoreV2:18*"
_class
loc:@pi/dense_2/bias*
use_locking(*
_output_shapes
:*
T0*
validate_shape(
�
save_11/Assign_19Assignpi/dense_2/kernelsave_11/RestoreV2:19*
use_locking(*
validate_shape(*
T0*$
_class
loc:@pi/dense_2/kernel*
_output_shapes

:@
�
save_11/Assign_20Assignpi/dense_2/kernel/Adamsave_11/RestoreV2:20*
validate_shape(*
T0*
_output_shapes

:@*
use_locking(*$
_class
loc:@pi/dense_2/kernel
�
save_11/Assign_21Assignpi/dense_2/kernel/Adam_1save_11/RestoreV2:21*
_output_shapes

:@*
use_locking(*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
T0
�
save_11/Assign_22Assignv/dense/biassave_11/RestoreV2:22*
_class
loc:@v/dense/bias*
_output_shapes
:@*
validate_shape(*
T0*
use_locking(
�
save_11/Assign_23Assignv/dense/bias/Adamsave_11/RestoreV2:23*
use_locking(*
_class
loc:@v/dense/bias*
validate_shape(*
T0*
_output_shapes
:@
�
save_11/Assign_24Assignv/dense/bias/Adam_1save_11/RestoreV2:24*
use_locking(*
_class
loc:@v/dense/bias*
T0*
_output_shapes
:@*
validate_shape(
�
save_11/Assign_25Assignv/dense/kernelsave_11/RestoreV2:25*
T0*
_output_shapes

:@*
validate_shape(*
use_locking(*!
_class
loc:@v/dense/kernel
�
save_11/Assign_26Assignv/dense/kernel/Adamsave_11/RestoreV2:26*!
_class
loc:@v/dense/kernel*
T0*
_output_shapes

:@*
validate_shape(*
use_locking(
�
save_11/Assign_27Assignv/dense/kernel/Adam_1save_11/RestoreV2:27*
use_locking(*!
_class
loc:@v/dense/kernel*
_output_shapes

:@*
T0*
validate_shape(
�
save_11/Assign_28Assignv/dense_1/biassave_11/RestoreV2:28*
_output_shapes
:@*!
_class
loc:@v/dense_1/bias*
T0*
validate_shape(*
use_locking(
�
save_11/Assign_29Assignv/dense_1/bias/Adamsave_11/RestoreV2:29*
_output_shapes
:@*
validate_shape(*
T0*
use_locking(*!
_class
loc:@v/dense_1/bias
�
save_11/Assign_30Assignv/dense_1/bias/Adam_1save_11/RestoreV2:30*
validate_shape(*
_output_shapes
:@*
T0*
use_locking(*!
_class
loc:@v/dense_1/bias
�
save_11/Assign_31Assignv/dense_1/kernelsave_11/RestoreV2:31*
validate_shape(*
_output_shapes

:@@*#
_class
loc:@v/dense_1/kernel*
use_locking(*
T0
�
save_11/Assign_32Assignv/dense_1/kernel/Adamsave_11/RestoreV2:32*
T0*#
_class
loc:@v/dense_1/kernel*
use_locking(*
validate_shape(*
_output_shapes

:@@
�
save_11/Assign_33Assignv/dense_1/kernel/Adam_1save_11/RestoreV2:33*
validate_shape(*
T0*
use_locking(*
_output_shapes

:@@*#
_class
loc:@v/dense_1/kernel
�
save_11/Assign_34Assignv/dense_2/biassave_11/RestoreV2:34*!
_class
loc:@v/dense_2/bias*
validate_shape(*
T0*
_output_shapes
:*
use_locking(
�
save_11/Assign_35Assignv/dense_2/bias/Adamsave_11/RestoreV2:35*
T0*
use_locking(*
validate_shape(*!
_class
loc:@v/dense_2/bias*
_output_shapes
:
�
save_11/Assign_36Assignv/dense_2/bias/Adam_1save_11/RestoreV2:36*
_output_shapes
:*
use_locking(*
validate_shape(*!
_class
loc:@v/dense_2/bias*
T0
�
save_11/Assign_37Assignv/dense_2/kernelsave_11/RestoreV2:37*
validate_shape(*
T0*#
_class
loc:@v/dense_2/kernel*
_output_shapes

:@*
use_locking(
�
save_11/Assign_38Assignv/dense_2/kernel/Adamsave_11/RestoreV2:38*
T0*
validate_shape(*#
_class
loc:@v/dense_2/kernel*
use_locking(*
_output_shapes

:@
�
save_11/Assign_39Assignv/dense_2/kernel/Adam_1save_11/RestoreV2:39*
validate_shape(*
T0*
use_locking(*#
_class
loc:@v/dense_2/kernel*
_output_shapes

:@
�
save_11/restore_shardNoOp^save_11/Assign^save_11/Assign_1^save_11/Assign_10^save_11/Assign_11^save_11/Assign_12^save_11/Assign_13^save_11/Assign_14^save_11/Assign_15^save_11/Assign_16^save_11/Assign_17^save_11/Assign_18^save_11/Assign_19^save_11/Assign_2^save_11/Assign_20^save_11/Assign_21^save_11/Assign_22^save_11/Assign_23^save_11/Assign_24^save_11/Assign_25^save_11/Assign_26^save_11/Assign_27^save_11/Assign_28^save_11/Assign_29^save_11/Assign_3^save_11/Assign_30^save_11/Assign_31^save_11/Assign_32^save_11/Assign_33^save_11/Assign_34^save_11/Assign_35^save_11/Assign_36^save_11/Assign_37^save_11/Assign_38^save_11/Assign_39^save_11/Assign_4^save_11/Assign_5^save_11/Assign_6^save_11/Assign_7^save_11/Assign_8^save_11/Assign_9
3
save_11/restore_allNoOp^save_11/restore_shard
\
save_12/filename/inputConst*
_output_shapes
: *
dtype0*
valueB Bmodel
t
save_12/filenamePlaceholderWithDefaultsave_12/filename/input*
_output_shapes
: *
dtype0*
shape: 
k
save_12/ConstPlaceholderWithDefaultsave_12/filename*
shape: *
dtype0*
_output_shapes
: 
�
save_12/StringJoin/inputs_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+_temp_aa8f329ad7be406d8898c6de552e7c86/part
~
save_12/StringJoin
StringJoinsave_12/Constsave_12/StringJoin/inputs_1*
_output_shapes
: *
N*
	separator 
T
save_12/num_shardsConst*
value	B :*
_output_shapes
: *
dtype0
_
save_12/ShardedFilename/shardConst*
value	B : *
_output_shapes
: *
dtype0
�
save_12/ShardedFilenameShardedFilenamesave_12/StringJoinsave_12/ShardedFilename/shardsave_12/num_shards*
_output_shapes
: 
�
save_12/SaveV2/tensor_namesConst*
_output_shapes
:(*
dtype0*�
value�B�(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1
�
save_12/SaveV2/shape_and_slicesConst*
_output_shapes
:(*
dtype0*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
�
save_12/SaveV2SaveV2save_12/ShardedFilenamesave_12/SaveV2/tensor_namessave_12/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1v/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1v/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1v/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1v/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1v/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1v/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1*6
dtypes,
*2(
�
save_12/control_dependencyIdentitysave_12/ShardedFilename^save_12/SaveV2*
T0**
_class 
loc:@save_12/ShardedFilename*
_output_shapes
: 
�
.save_12/MergeV2Checkpoints/checkpoint_prefixesPacksave_12/ShardedFilename^save_12/control_dependency*
T0*

axis *
_output_shapes
:*
N
�
save_12/MergeV2CheckpointsMergeV2Checkpoints.save_12/MergeV2Checkpoints/checkpoint_prefixessave_12/Const*
delete_old_dirs(
�
save_12/IdentityIdentitysave_12/Const^save_12/MergeV2Checkpoints^save_12/control_dependency*
T0*
_output_shapes
: 
�
save_12/RestoreV2/tensor_namesConst*
dtype0*�
value�B�(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
_output_shapes
:(
�
"save_12/RestoreV2/shape_and_slicesConst*
_output_shapes
:(*
dtype0*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
�
save_12/RestoreV2	RestoreV2save_12/Constsave_12/RestoreV2/tensor_names"save_12/RestoreV2/shape_and_slices*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::*6
dtypes,
*2(
�
save_12/AssignAssignbeta1_powersave_12/RestoreV2*
use_locking(* 
_class
loc:@pi/dense/bias*
_output_shapes
: *
validate_shape(*
T0
�
save_12/Assign_1Assignbeta1_power_1save_12/RestoreV2:1*
use_locking(*
_output_shapes
: *
validate_shape(*
_class
loc:@v/dense/bias*
T0
�
save_12/Assign_2Assignbeta2_powersave_12/RestoreV2:2*
T0*
_output_shapes
: *
validate_shape(*
use_locking(* 
_class
loc:@pi/dense/bias
�
save_12/Assign_3Assignbeta2_power_1save_12/RestoreV2:3*
T0*
validate_shape(*
_class
loc:@v/dense/bias*
use_locking(*
_output_shapes
: 
�
save_12/Assign_4Assignpi/dense/biassave_12/RestoreV2:4*
use_locking(*
T0*
validate_shape(* 
_class
loc:@pi/dense/bias*
_output_shapes
:@
�
save_12/Assign_5Assignpi/dense/bias/Adamsave_12/RestoreV2:5* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
:@*
T0*
use_locking(
�
save_12/Assign_6Assignpi/dense/bias/Adam_1save_12/RestoreV2:6*
_output_shapes
:@*
validate_shape(* 
_class
loc:@pi/dense/bias*
use_locking(*
T0
�
save_12/Assign_7Assignpi/dense/kernelsave_12/RestoreV2:7*
use_locking(*
validate_shape(*
_output_shapes

:@*"
_class
loc:@pi/dense/kernel*
T0
�
save_12/Assign_8Assignpi/dense/kernel/Adamsave_12/RestoreV2:8*
use_locking(*"
_class
loc:@pi/dense/kernel*
validate_shape(*
T0*
_output_shapes

:@
�
save_12/Assign_9Assignpi/dense/kernel/Adam_1save_12/RestoreV2:9*
_output_shapes

:@*
validate_shape(*
use_locking(*"
_class
loc:@pi/dense/kernel*
T0
�
save_12/Assign_10Assignpi/dense_1/biassave_12/RestoreV2:10*
validate_shape(*"
_class
loc:@pi/dense_1/bias*
_output_shapes
:@*
use_locking(*
T0
�
save_12/Assign_11Assignpi/dense_1/bias/Adamsave_12/RestoreV2:11*
T0*"
_class
loc:@pi/dense_1/bias*
use_locking(*
validate_shape(*
_output_shapes
:@
�
save_12/Assign_12Assignpi/dense_1/bias/Adam_1save_12/RestoreV2:12*
T0*
validate_shape(*
_output_shapes
:@*"
_class
loc:@pi/dense_1/bias*
use_locking(
�
save_12/Assign_13Assignpi/dense_1/kernelsave_12/RestoreV2:13*
T0*
_output_shapes

:@@*$
_class
loc:@pi/dense_1/kernel*
validate_shape(*
use_locking(
�
save_12/Assign_14Assignpi/dense_1/kernel/Adamsave_12/RestoreV2:14*
use_locking(*
_output_shapes

:@@*
validate_shape(*
T0*$
_class
loc:@pi/dense_1/kernel
�
save_12/Assign_15Assignpi/dense_1/kernel/Adam_1save_12/RestoreV2:15*$
_class
loc:@pi/dense_1/kernel*
use_locking(*
validate_shape(*
T0*
_output_shapes

:@@
�
save_12/Assign_16Assignpi/dense_2/biassave_12/RestoreV2:16*
use_locking(*
_output_shapes
:*
validate_shape(*"
_class
loc:@pi/dense_2/bias*
T0
�
save_12/Assign_17Assignpi/dense_2/bias/Adamsave_12/RestoreV2:17*
validate_shape(*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:
�
save_12/Assign_18Assignpi/dense_2/bias/Adam_1save_12/RestoreV2:18*
validate_shape(*
T0*"
_class
loc:@pi/dense_2/bias*
use_locking(*
_output_shapes
:
�
save_12/Assign_19Assignpi/dense_2/kernelsave_12/RestoreV2:19*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0*$
_class
loc:@pi/dense_2/kernel
�
save_12/Assign_20Assignpi/dense_2/kernel/Adamsave_12/RestoreV2:20*
T0*
_output_shapes

:@*
validate_shape(*
use_locking(*$
_class
loc:@pi/dense_2/kernel
�
save_12/Assign_21Assignpi/dense_2/kernel/Adam_1save_12/RestoreV2:21*
_output_shapes

:@*
use_locking(*
validate_shape(*
T0*$
_class
loc:@pi/dense_2/kernel
�
save_12/Assign_22Assignv/dense/biassave_12/RestoreV2:22*
validate_shape(*
_output_shapes
:@*
_class
loc:@v/dense/bias*
use_locking(*
T0
�
save_12/Assign_23Assignv/dense/bias/Adamsave_12/RestoreV2:23*
use_locking(*
_output_shapes
:@*
_class
loc:@v/dense/bias*
T0*
validate_shape(
�
save_12/Assign_24Assignv/dense/bias/Adam_1save_12/RestoreV2:24*
_output_shapes
:@*
T0*
use_locking(*
_class
loc:@v/dense/bias*
validate_shape(
�
save_12/Assign_25Assignv/dense/kernelsave_12/RestoreV2:25*!
_class
loc:@v/dense/kernel*
use_locking(*
_output_shapes

:@*
T0*
validate_shape(
�
save_12/Assign_26Assignv/dense/kernel/Adamsave_12/RestoreV2:26*
_output_shapes

:@*
T0*
use_locking(*
validate_shape(*!
_class
loc:@v/dense/kernel
�
save_12/Assign_27Assignv/dense/kernel/Adam_1save_12/RestoreV2:27*
validate_shape(*!
_class
loc:@v/dense/kernel*
T0*
_output_shapes

:@*
use_locking(
�
save_12/Assign_28Assignv/dense_1/biassave_12/RestoreV2:28*
validate_shape(*
_output_shapes
:@*
T0*
use_locking(*!
_class
loc:@v/dense_1/bias
�
save_12/Assign_29Assignv/dense_1/bias/Adamsave_12/RestoreV2:29*!
_class
loc:@v/dense_1/bias*
_output_shapes
:@*
use_locking(*
validate_shape(*
T0
�
save_12/Assign_30Assignv/dense_1/bias/Adam_1save_12/RestoreV2:30*
T0*
use_locking(*
validate_shape(*!
_class
loc:@v/dense_1/bias*
_output_shapes
:@
�
save_12/Assign_31Assignv/dense_1/kernelsave_12/RestoreV2:31*
use_locking(*
_output_shapes

:@@*
validate_shape(*
T0*#
_class
loc:@v/dense_1/kernel
�
save_12/Assign_32Assignv/dense_1/kernel/Adamsave_12/RestoreV2:32*#
_class
loc:@v/dense_1/kernel*
T0*
validate_shape(*
_output_shapes

:@@*
use_locking(
�
save_12/Assign_33Assignv/dense_1/kernel/Adam_1save_12/RestoreV2:33*
validate_shape(*#
_class
loc:@v/dense_1/kernel*
_output_shapes

:@@*
T0*
use_locking(
�
save_12/Assign_34Assignv/dense_2/biassave_12/RestoreV2:34*
T0*!
_class
loc:@v/dense_2/bias*
validate_shape(*
use_locking(*
_output_shapes
:
�
save_12/Assign_35Assignv/dense_2/bias/Adamsave_12/RestoreV2:35*
_output_shapes
:*
T0*
use_locking(*!
_class
loc:@v/dense_2/bias*
validate_shape(
�
save_12/Assign_36Assignv/dense_2/bias/Adam_1save_12/RestoreV2:36*
_output_shapes
:*
T0*
validate_shape(*
use_locking(*!
_class
loc:@v/dense_2/bias
�
save_12/Assign_37Assignv/dense_2/kernelsave_12/RestoreV2:37*
validate_shape(*
_output_shapes

:@*
T0*
use_locking(*#
_class
loc:@v/dense_2/kernel
�
save_12/Assign_38Assignv/dense_2/kernel/Adamsave_12/RestoreV2:38*
_output_shapes

:@*
T0*
use_locking(*
validate_shape(*#
_class
loc:@v/dense_2/kernel
�
save_12/Assign_39Assignv/dense_2/kernel/Adam_1save_12/RestoreV2:39*
use_locking(*
_output_shapes

:@*
validate_shape(*
T0*#
_class
loc:@v/dense_2/kernel
�
save_12/restore_shardNoOp^save_12/Assign^save_12/Assign_1^save_12/Assign_10^save_12/Assign_11^save_12/Assign_12^save_12/Assign_13^save_12/Assign_14^save_12/Assign_15^save_12/Assign_16^save_12/Assign_17^save_12/Assign_18^save_12/Assign_19^save_12/Assign_2^save_12/Assign_20^save_12/Assign_21^save_12/Assign_22^save_12/Assign_23^save_12/Assign_24^save_12/Assign_25^save_12/Assign_26^save_12/Assign_27^save_12/Assign_28^save_12/Assign_29^save_12/Assign_3^save_12/Assign_30^save_12/Assign_31^save_12/Assign_32^save_12/Assign_33^save_12/Assign_34^save_12/Assign_35^save_12/Assign_36^save_12/Assign_37^save_12/Assign_38^save_12/Assign_39^save_12/Assign_4^save_12/Assign_5^save_12/Assign_6^save_12/Assign_7^save_12/Assign_8^save_12/Assign_9
3
save_12/restore_allNoOp^save_12/restore_shard
\
save_13/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
t
save_13/filenamePlaceholderWithDefaultsave_13/filename/input*
shape: *
_output_shapes
: *
dtype0
k
save_13/ConstPlaceholderWithDefaultsave_13/filename*
shape: *
dtype0*
_output_shapes
: 
�
save_13/StringJoin/inputs_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+_temp_360ee69df0f94f418a8e3798ee73f0f0/part
~
save_13/StringJoin
StringJoinsave_13/Constsave_13/StringJoin/inputs_1*
_output_shapes
: *
	separator *
N
T
save_13/num_shardsConst*
dtype0*
_output_shapes
: *
value	B :
_
save_13/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 
�
save_13/ShardedFilenameShardedFilenamesave_13/StringJoinsave_13/ShardedFilename/shardsave_13/num_shards*
_output_shapes
: 
�
save_13/SaveV2/tensor_namesConst*
_output_shapes
:(*
dtype0*�
value�B�(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1
�
save_13/SaveV2/shape_and_slicesConst*
dtype0*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:(
�
save_13/SaveV2SaveV2save_13/ShardedFilenamesave_13/SaveV2/tensor_namessave_13/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1v/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1v/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1v/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1v/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1v/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1v/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1*6
dtypes,
*2(
�
save_13/control_dependencyIdentitysave_13/ShardedFilename^save_13/SaveV2**
_class 
loc:@save_13/ShardedFilename*
_output_shapes
: *
T0
�
.save_13/MergeV2Checkpoints/checkpoint_prefixesPacksave_13/ShardedFilename^save_13/control_dependency*
N*

axis *
T0*
_output_shapes
:
�
save_13/MergeV2CheckpointsMergeV2Checkpoints.save_13/MergeV2Checkpoints/checkpoint_prefixessave_13/Const*
delete_old_dirs(
�
save_13/IdentityIdentitysave_13/Const^save_13/MergeV2Checkpoints^save_13/control_dependency*
_output_shapes
: *
T0
�
save_13/RestoreV2/tensor_namesConst*
_output_shapes
:(*�
value�B�(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
dtype0
�
"save_13/RestoreV2/shape_and_slicesConst*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:(*
dtype0
�
save_13/RestoreV2	RestoreV2save_13/Constsave_13/RestoreV2/tensor_names"save_13/RestoreV2/shape_and_slices*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::*6
dtypes,
*2(
�
save_13/AssignAssignbeta1_powersave_13/RestoreV2*
_output_shapes
: * 
_class
loc:@pi/dense/bias*
T0*
use_locking(*
validate_shape(
�
save_13/Assign_1Assignbeta1_power_1save_13/RestoreV2:1*
use_locking(*
_output_shapes
: *
_class
loc:@v/dense/bias*
T0*
validate_shape(
�
save_13/Assign_2Assignbeta2_powersave_13/RestoreV2:2* 
_class
loc:@pi/dense/bias*
validate_shape(*
use_locking(*
T0*
_output_shapes
: 
�
save_13/Assign_3Assignbeta2_power_1save_13/RestoreV2:3*
_output_shapes
: *
validate_shape(*
use_locking(*
_class
loc:@v/dense/bias*
T0
�
save_13/Assign_4Assignpi/dense/biassave_13/RestoreV2:4*
_output_shapes
:@*
use_locking(* 
_class
loc:@pi/dense/bias*
T0*
validate_shape(
�
save_13/Assign_5Assignpi/dense/bias/Adamsave_13/RestoreV2:5*
use_locking(* 
_class
loc:@pi/dense/bias*
T0*
_output_shapes
:@*
validate_shape(
�
save_13/Assign_6Assignpi/dense/bias/Adam_1save_13/RestoreV2:6*
validate_shape(* 
_class
loc:@pi/dense/bias*
_output_shapes
:@*
T0*
use_locking(
�
save_13/Assign_7Assignpi/dense/kernelsave_13/RestoreV2:7*"
_class
loc:@pi/dense/kernel*
T0*
validate_shape(*
_output_shapes

:@*
use_locking(
�
save_13/Assign_8Assignpi/dense/kernel/Adamsave_13/RestoreV2:8*
use_locking(*
_output_shapes

:@*
validate_shape(*
T0*"
_class
loc:@pi/dense/kernel
�
save_13/Assign_9Assignpi/dense/kernel/Adam_1save_13/RestoreV2:9*
T0*
validate_shape(*"
_class
loc:@pi/dense/kernel*
use_locking(*
_output_shapes

:@
�
save_13/Assign_10Assignpi/dense_1/biassave_13/RestoreV2:10*"
_class
loc:@pi/dense_1/bias*
_output_shapes
:@*
use_locking(*
T0*
validate_shape(
�
save_13/Assign_11Assignpi/dense_1/bias/Adamsave_13/RestoreV2:11*
T0*
validate_shape(*
_output_shapes
:@*"
_class
loc:@pi/dense_1/bias*
use_locking(
�
save_13/Assign_12Assignpi/dense_1/bias/Adam_1save_13/RestoreV2:12*
T0*
validate_shape(*
_output_shapes
:@*
use_locking(*"
_class
loc:@pi/dense_1/bias
�
save_13/Assign_13Assignpi/dense_1/kernelsave_13/RestoreV2:13*
use_locking(*
_output_shapes

:@@*$
_class
loc:@pi/dense_1/kernel*
validate_shape(*
T0
�
save_13/Assign_14Assignpi/dense_1/kernel/Adamsave_13/RestoreV2:14*$
_class
loc:@pi/dense_1/kernel*
_output_shapes

:@@*
validate_shape(*
T0*
use_locking(
�
save_13/Assign_15Assignpi/dense_1/kernel/Adam_1save_13/RestoreV2:15*$
_class
loc:@pi/dense_1/kernel*
use_locking(*
_output_shapes

:@@*
T0*
validate_shape(
�
save_13/Assign_16Assignpi/dense_2/biassave_13/RestoreV2:16*"
_class
loc:@pi/dense_2/bias*
T0*
validate_shape(*
use_locking(*
_output_shapes
:
�
save_13/Assign_17Assignpi/dense_2/bias/Adamsave_13/RestoreV2:17*
validate_shape(*
_output_shapes
:*
T0*"
_class
loc:@pi/dense_2/bias*
use_locking(
�
save_13/Assign_18Assignpi/dense_2/bias/Adam_1save_13/RestoreV2:18*
_output_shapes
:*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
T0*
use_locking(
�
save_13/Assign_19Assignpi/dense_2/kernelsave_13/RestoreV2:19*$
_class
loc:@pi/dense_2/kernel*
T0*
use_locking(*
_output_shapes

:@*
validate_shape(
�
save_13/Assign_20Assignpi/dense_2/kernel/Adamsave_13/RestoreV2:20*$
_class
loc:@pi/dense_2/kernel*
_output_shapes

:@*
use_locking(*
validate_shape(*
T0
�
save_13/Assign_21Assignpi/dense_2/kernel/Adam_1save_13/RestoreV2:21*
T0*$
_class
loc:@pi/dense_2/kernel*
_output_shapes

:@*
use_locking(*
validate_shape(
�
save_13/Assign_22Assignv/dense/biassave_13/RestoreV2:22*
use_locking(*
validate_shape(*
_output_shapes
:@*
_class
loc:@v/dense/bias*
T0
�
save_13/Assign_23Assignv/dense/bias/Adamsave_13/RestoreV2:23*
_output_shapes
:@*
_class
loc:@v/dense/bias*
T0*
validate_shape(*
use_locking(
�
save_13/Assign_24Assignv/dense/bias/Adam_1save_13/RestoreV2:24*
use_locking(*
_output_shapes
:@*
T0*
validate_shape(*
_class
loc:@v/dense/bias
�
save_13/Assign_25Assignv/dense/kernelsave_13/RestoreV2:25*!
_class
loc:@v/dense/kernel*
_output_shapes

:@*
validate_shape(*
T0*
use_locking(
�
save_13/Assign_26Assignv/dense/kernel/Adamsave_13/RestoreV2:26*!
_class
loc:@v/dense/kernel*
validate_shape(*
T0*
_output_shapes

:@*
use_locking(
�
save_13/Assign_27Assignv/dense/kernel/Adam_1save_13/RestoreV2:27*
T0*
_output_shapes

:@*
use_locking(*!
_class
loc:@v/dense/kernel*
validate_shape(
�
save_13/Assign_28Assignv/dense_1/biassave_13/RestoreV2:28*!
_class
loc:@v/dense_1/bias*
_output_shapes
:@*
validate_shape(*
T0*
use_locking(
�
save_13/Assign_29Assignv/dense_1/bias/Adamsave_13/RestoreV2:29*
use_locking(*
T0*!
_class
loc:@v/dense_1/bias*
_output_shapes
:@*
validate_shape(
�
save_13/Assign_30Assignv/dense_1/bias/Adam_1save_13/RestoreV2:30*
T0*!
_class
loc:@v/dense_1/bias*
use_locking(*
_output_shapes
:@*
validate_shape(
�
save_13/Assign_31Assignv/dense_1/kernelsave_13/RestoreV2:31*
use_locking(*#
_class
loc:@v/dense_1/kernel*
T0*
validate_shape(*
_output_shapes

:@@
�
save_13/Assign_32Assignv/dense_1/kernel/Adamsave_13/RestoreV2:32*
use_locking(*
validate_shape(*
_output_shapes

:@@*
T0*#
_class
loc:@v/dense_1/kernel
�
save_13/Assign_33Assignv/dense_1/kernel/Adam_1save_13/RestoreV2:33*
T0*#
_class
loc:@v/dense_1/kernel*
validate_shape(*
_output_shapes

:@@*
use_locking(
�
save_13/Assign_34Assignv/dense_2/biassave_13/RestoreV2:34*!
_class
loc:@v/dense_2/bias*
_output_shapes
:*
validate_shape(*
use_locking(*
T0
�
save_13/Assign_35Assignv/dense_2/bias/Adamsave_13/RestoreV2:35*!
_class
loc:@v/dense_2/bias*
T0*
_output_shapes
:*
validate_shape(*
use_locking(
�
save_13/Assign_36Assignv/dense_2/bias/Adam_1save_13/RestoreV2:36*
use_locking(*
validate_shape(*
_output_shapes
:*
T0*!
_class
loc:@v/dense_2/bias
�
save_13/Assign_37Assignv/dense_2/kernelsave_13/RestoreV2:37*
T0*
use_locking(*
validate_shape(*#
_class
loc:@v/dense_2/kernel*
_output_shapes

:@
�
save_13/Assign_38Assignv/dense_2/kernel/Adamsave_13/RestoreV2:38*#
_class
loc:@v/dense_2/kernel*
use_locking(*
_output_shapes

:@*
validate_shape(*
T0
�
save_13/Assign_39Assignv/dense_2/kernel/Adam_1save_13/RestoreV2:39*#
_class
loc:@v/dense_2/kernel*
validate_shape(*
use_locking(*
T0*
_output_shapes

:@
�
save_13/restore_shardNoOp^save_13/Assign^save_13/Assign_1^save_13/Assign_10^save_13/Assign_11^save_13/Assign_12^save_13/Assign_13^save_13/Assign_14^save_13/Assign_15^save_13/Assign_16^save_13/Assign_17^save_13/Assign_18^save_13/Assign_19^save_13/Assign_2^save_13/Assign_20^save_13/Assign_21^save_13/Assign_22^save_13/Assign_23^save_13/Assign_24^save_13/Assign_25^save_13/Assign_26^save_13/Assign_27^save_13/Assign_28^save_13/Assign_29^save_13/Assign_3^save_13/Assign_30^save_13/Assign_31^save_13/Assign_32^save_13/Assign_33^save_13/Assign_34^save_13/Assign_35^save_13/Assign_36^save_13/Assign_37^save_13/Assign_38^save_13/Assign_39^save_13/Assign_4^save_13/Assign_5^save_13/Assign_6^save_13/Assign_7^save_13/Assign_8^save_13/Assign_9
3
save_13/restore_allNoOp^save_13/restore_shard
\
save_14/filename/inputConst*
dtype0*
valueB Bmodel*
_output_shapes
: 
t
save_14/filenamePlaceholderWithDefaultsave_14/filename/input*
dtype0*
shape: *
_output_shapes
: 
k
save_14/ConstPlaceholderWithDefaultsave_14/filename*
_output_shapes
: *
shape: *
dtype0
�
save_14/StringJoin/inputs_1Const*
_output_shapes
: *<
value3B1 B+_temp_e383e4e4456e4c038604994516d88918/part*
dtype0
~
save_14/StringJoin
StringJoinsave_14/Constsave_14/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
T
save_14/num_shardsConst*
_output_shapes
: *
value	B :*
dtype0
_
save_14/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 
�
save_14/ShardedFilenameShardedFilenamesave_14/StringJoinsave_14/ShardedFilename/shardsave_14/num_shards*
_output_shapes
: 
�
save_14/SaveV2/tensor_namesConst*
_output_shapes
:(*
dtype0*�
value�B�(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1
�
save_14/SaveV2/shape_and_slicesConst*
_output_shapes
:(*
dtype0*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
�
save_14/SaveV2SaveV2save_14/ShardedFilenamesave_14/SaveV2/tensor_namessave_14/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1v/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1v/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1v/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1v/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1v/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1v/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1*6
dtypes,
*2(
�
save_14/control_dependencyIdentitysave_14/ShardedFilename^save_14/SaveV2*
_output_shapes
: **
_class 
loc:@save_14/ShardedFilename*
T0
�
.save_14/MergeV2Checkpoints/checkpoint_prefixesPacksave_14/ShardedFilename^save_14/control_dependency*
_output_shapes
:*
T0*
N*

axis 
�
save_14/MergeV2CheckpointsMergeV2Checkpoints.save_14/MergeV2Checkpoints/checkpoint_prefixessave_14/Const*
delete_old_dirs(
�
save_14/IdentityIdentitysave_14/Const^save_14/MergeV2Checkpoints^save_14/control_dependency*
_output_shapes
: *
T0
�
save_14/RestoreV2/tensor_namesConst*
_output_shapes
:(*
dtype0*�
value�B�(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1
�
"save_14/RestoreV2/shape_and_slicesConst*
dtype0*
_output_shapes
:(*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
�
save_14/RestoreV2	RestoreV2save_14/Constsave_14/RestoreV2/tensor_names"save_14/RestoreV2/shape_and_slices*6
dtypes,
*2(*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::
�
save_14/AssignAssignbeta1_powersave_14/RestoreV2* 
_class
loc:@pi/dense/bias*
use_locking(*
_output_shapes
: *
T0*
validate_shape(
�
save_14/Assign_1Assignbeta1_power_1save_14/RestoreV2:1*
_output_shapes
: *
_class
loc:@v/dense/bias*
T0*
use_locking(*
validate_shape(
�
save_14/Assign_2Assignbeta2_powersave_14/RestoreV2:2*
validate_shape(*
_output_shapes
: *
T0*
use_locking(* 
_class
loc:@pi/dense/bias
�
save_14/Assign_3Assignbeta2_power_1save_14/RestoreV2:3*
validate_shape(*
_class
loc:@v/dense/bias*
_output_shapes
: *
T0*
use_locking(
�
save_14/Assign_4Assignpi/dense/biassave_14/RestoreV2:4*
validate_shape(* 
_class
loc:@pi/dense/bias*
_output_shapes
:@*
use_locking(*
T0
�
save_14/Assign_5Assignpi/dense/bias/Adamsave_14/RestoreV2:5* 
_class
loc:@pi/dense/bias*
T0*
_output_shapes
:@*
use_locking(*
validate_shape(
�
save_14/Assign_6Assignpi/dense/bias/Adam_1save_14/RestoreV2:6*
validate_shape(*
T0*
_output_shapes
:@*
use_locking(* 
_class
loc:@pi/dense/bias
�
save_14/Assign_7Assignpi/dense/kernelsave_14/RestoreV2:7*"
_class
loc:@pi/dense/kernel*
use_locking(*
T0*
_output_shapes

:@*
validate_shape(
�
save_14/Assign_8Assignpi/dense/kernel/Adamsave_14/RestoreV2:8*"
_class
loc:@pi/dense/kernel*
validate_shape(*
use_locking(*
T0*
_output_shapes

:@
�
save_14/Assign_9Assignpi/dense/kernel/Adam_1save_14/RestoreV2:9*"
_class
loc:@pi/dense/kernel*
T0*
validate_shape(*
_output_shapes

:@*
use_locking(
�
save_14/Assign_10Assignpi/dense_1/biassave_14/RestoreV2:10*
validate_shape(*
T0*
use_locking(*"
_class
loc:@pi/dense_1/bias*
_output_shapes
:@
�
save_14/Assign_11Assignpi/dense_1/bias/Adamsave_14/RestoreV2:11*
use_locking(*
_output_shapes
:@*
T0*"
_class
loc:@pi/dense_1/bias*
validate_shape(
�
save_14/Assign_12Assignpi/dense_1/bias/Adam_1save_14/RestoreV2:12*
T0*
validate_shape(*
_output_shapes
:@*"
_class
loc:@pi/dense_1/bias*
use_locking(
�
save_14/Assign_13Assignpi/dense_1/kernelsave_14/RestoreV2:13*$
_class
loc:@pi/dense_1/kernel*
use_locking(*
_output_shapes

:@@*
validate_shape(*
T0
�
save_14/Assign_14Assignpi/dense_1/kernel/Adamsave_14/RestoreV2:14*$
_class
loc:@pi/dense_1/kernel*
validate_shape(*
T0*
use_locking(*
_output_shapes

:@@
�
save_14/Assign_15Assignpi/dense_1/kernel/Adam_1save_14/RestoreV2:15*
T0*
validate_shape(*
_output_shapes

:@@*$
_class
loc:@pi/dense_1/kernel*
use_locking(
�
save_14/Assign_16Assignpi/dense_2/biassave_14/RestoreV2:16*
use_locking(*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
_output_shapes
:*
T0
�
save_14/Assign_17Assignpi/dense_2/bias/Adamsave_14/RestoreV2:17*
T0*
use_locking(*
validate_shape(*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:
�
save_14/Assign_18Assignpi/dense_2/bias/Adam_1save_14/RestoreV2:18*
_output_shapes
:*
T0*"
_class
loc:@pi/dense_2/bias*
use_locking(*
validate_shape(
�
save_14/Assign_19Assignpi/dense_2/kernelsave_14/RestoreV2:19*
_output_shapes

:@*
use_locking(*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
T0
�
save_14/Assign_20Assignpi/dense_2/kernel/Adamsave_14/RestoreV2:20*
T0*
validate_shape(*
use_locking(*
_output_shapes

:@*$
_class
loc:@pi/dense_2/kernel
�
save_14/Assign_21Assignpi/dense_2/kernel/Adam_1save_14/RestoreV2:21*
use_locking(*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
T0*
_output_shapes

:@
�
save_14/Assign_22Assignv/dense/biassave_14/RestoreV2:22*
validate_shape(*
_output_shapes
:@*
T0*
use_locking(*
_class
loc:@v/dense/bias
�
save_14/Assign_23Assignv/dense/bias/Adamsave_14/RestoreV2:23*
T0*
validate_shape(*
_class
loc:@v/dense/bias*
use_locking(*
_output_shapes
:@
�
save_14/Assign_24Assignv/dense/bias/Adam_1save_14/RestoreV2:24*
validate_shape(*
use_locking(*
_output_shapes
:@*
T0*
_class
loc:@v/dense/bias
�
save_14/Assign_25Assignv/dense/kernelsave_14/RestoreV2:25*
T0*
validate_shape(*
_output_shapes

:@*!
_class
loc:@v/dense/kernel*
use_locking(
�
save_14/Assign_26Assignv/dense/kernel/Adamsave_14/RestoreV2:26*
_output_shapes

:@*
T0*
validate_shape(*
use_locking(*!
_class
loc:@v/dense/kernel
�
save_14/Assign_27Assignv/dense/kernel/Adam_1save_14/RestoreV2:27*
validate_shape(*
T0*!
_class
loc:@v/dense/kernel*
_output_shapes

:@*
use_locking(
�
save_14/Assign_28Assignv/dense_1/biassave_14/RestoreV2:28*
use_locking(*!
_class
loc:@v/dense_1/bias*
validate_shape(*
T0*
_output_shapes
:@
�
save_14/Assign_29Assignv/dense_1/bias/Adamsave_14/RestoreV2:29*
_output_shapes
:@*
validate_shape(*
T0*!
_class
loc:@v/dense_1/bias*
use_locking(
�
save_14/Assign_30Assignv/dense_1/bias/Adam_1save_14/RestoreV2:30*
validate_shape(*
use_locking(*
T0*!
_class
loc:@v/dense_1/bias*
_output_shapes
:@
�
save_14/Assign_31Assignv/dense_1/kernelsave_14/RestoreV2:31*
T0*
validate_shape(*
use_locking(*
_output_shapes

:@@*#
_class
loc:@v/dense_1/kernel
�
save_14/Assign_32Assignv/dense_1/kernel/Adamsave_14/RestoreV2:32*#
_class
loc:@v/dense_1/kernel*
T0*
use_locking(*
_output_shapes

:@@*
validate_shape(
�
save_14/Assign_33Assignv/dense_1/kernel/Adam_1save_14/RestoreV2:33*
validate_shape(*
use_locking(*
T0*
_output_shapes

:@@*#
_class
loc:@v/dense_1/kernel
�
save_14/Assign_34Assignv/dense_2/biassave_14/RestoreV2:34*!
_class
loc:@v/dense_2/bias*
T0*
validate_shape(*
_output_shapes
:*
use_locking(
�
save_14/Assign_35Assignv/dense_2/bias/Adamsave_14/RestoreV2:35*
use_locking(*
T0*!
_class
loc:@v/dense_2/bias*
validate_shape(*
_output_shapes
:
�
save_14/Assign_36Assignv/dense_2/bias/Adam_1save_14/RestoreV2:36*
T0*
_output_shapes
:*!
_class
loc:@v/dense_2/bias*
use_locking(*
validate_shape(
�
save_14/Assign_37Assignv/dense_2/kernelsave_14/RestoreV2:37*#
_class
loc:@v/dense_2/kernel*
validate_shape(*
T0*
_output_shapes

:@*
use_locking(
�
save_14/Assign_38Assignv/dense_2/kernel/Adamsave_14/RestoreV2:38*
_output_shapes

:@*
validate_shape(*
use_locking(*#
_class
loc:@v/dense_2/kernel*
T0
�
save_14/Assign_39Assignv/dense_2/kernel/Adam_1save_14/RestoreV2:39*#
_class
loc:@v/dense_2/kernel*
_output_shapes

:@*
validate_shape(*
T0*
use_locking(
�
save_14/restore_shardNoOp^save_14/Assign^save_14/Assign_1^save_14/Assign_10^save_14/Assign_11^save_14/Assign_12^save_14/Assign_13^save_14/Assign_14^save_14/Assign_15^save_14/Assign_16^save_14/Assign_17^save_14/Assign_18^save_14/Assign_19^save_14/Assign_2^save_14/Assign_20^save_14/Assign_21^save_14/Assign_22^save_14/Assign_23^save_14/Assign_24^save_14/Assign_25^save_14/Assign_26^save_14/Assign_27^save_14/Assign_28^save_14/Assign_29^save_14/Assign_3^save_14/Assign_30^save_14/Assign_31^save_14/Assign_32^save_14/Assign_33^save_14/Assign_34^save_14/Assign_35^save_14/Assign_36^save_14/Assign_37^save_14/Assign_38^save_14/Assign_39^save_14/Assign_4^save_14/Assign_5^save_14/Assign_6^save_14/Assign_7^save_14/Assign_8^save_14/Assign_9
3
save_14/restore_allNoOp^save_14/restore_shard
\
save_15/filename/inputConst*
valueB Bmodel*
_output_shapes
: *
dtype0
t
save_15/filenamePlaceholderWithDefaultsave_15/filename/input*
_output_shapes
: *
dtype0*
shape: 
k
save_15/ConstPlaceholderWithDefaultsave_15/filename*
_output_shapes
: *
shape: *
dtype0
�
save_15/StringJoin/inputs_1Const*
dtype0*<
value3B1 B+_temp_e06cbce75be14a33bc37e03f9f8ef7e3/part*
_output_shapes
: 
~
save_15/StringJoin
StringJoinsave_15/Constsave_15/StringJoin/inputs_1*
_output_shapes
: *
N*
	separator 
T
save_15/num_shardsConst*
_output_shapes
: *
value	B :*
dtype0
_
save_15/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 
�
save_15/ShardedFilenameShardedFilenamesave_15/StringJoinsave_15/ShardedFilename/shardsave_15/num_shards*
_output_shapes
: 
�
save_15/SaveV2/tensor_namesConst*�
value�B�(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:(
�
save_15/SaveV2/shape_and_slicesConst*
_output_shapes
:(*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
�
save_15/SaveV2SaveV2save_15/ShardedFilenamesave_15/SaveV2/tensor_namessave_15/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1v/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1v/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1v/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1v/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1v/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1v/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1*6
dtypes,
*2(
�
save_15/control_dependencyIdentitysave_15/ShardedFilename^save_15/SaveV2**
_class 
loc:@save_15/ShardedFilename*
_output_shapes
: *
T0
�
.save_15/MergeV2Checkpoints/checkpoint_prefixesPacksave_15/ShardedFilename^save_15/control_dependency*
_output_shapes
:*

axis *
N*
T0
�
save_15/MergeV2CheckpointsMergeV2Checkpoints.save_15/MergeV2Checkpoints/checkpoint_prefixessave_15/Const*
delete_old_dirs(
�
save_15/IdentityIdentitysave_15/Const^save_15/MergeV2Checkpoints^save_15/control_dependency*
_output_shapes
: *
T0
�
save_15/RestoreV2/tensor_namesConst*�
value�B�(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:(
�
"save_15/RestoreV2/shape_and_slicesConst*
dtype0*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:(
�
save_15/RestoreV2	RestoreV2save_15/Constsave_15/RestoreV2/tensor_names"save_15/RestoreV2/shape_and_slices*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::*6
dtypes,
*2(
�
save_15/AssignAssignbeta1_powersave_15/RestoreV2*
validate_shape(*
use_locking(*
_output_shapes
: * 
_class
loc:@pi/dense/bias*
T0
�
save_15/Assign_1Assignbeta1_power_1save_15/RestoreV2:1*
_class
loc:@v/dense/bias*
use_locking(*
_output_shapes
: *
validate_shape(*
T0
�
save_15/Assign_2Assignbeta2_powersave_15/RestoreV2:2*
use_locking(*
validate_shape(*
_output_shapes
: *
T0* 
_class
loc:@pi/dense/bias
�
save_15/Assign_3Assignbeta2_power_1save_15/RestoreV2:3*
T0*
_output_shapes
: *
validate_shape(*
_class
loc:@v/dense/bias*
use_locking(
�
save_15/Assign_4Assignpi/dense/biassave_15/RestoreV2:4*
T0*
use_locking(*
validate_shape(*
_output_shapes
:@* 
_class
loc:@pi/dense/bias
�
save_15/Assign_5Assignpi/dense/bias/Adamsave_15/RestoreV2:5*
_output_shapes
:@*
use_locking(* 
_class
loc:@pi/dense/bias*
T0*
validate_shape(
�
save_15/Assign_6Assignpi/dense/bias/Adam_1save_15/RestoreV2:6*
use_locking(*
_output_shapes
:@* 
_class
loc:@pi/dense/bias*
T0*
validate_shape(
�
save_15/Assign_7Assignpi/dense/kernelsave_15/RestoreV2:7*
use_locking(*
validate_shape(*"
_class
loc:@pi/dense/kernel*
_output_shapes

:@*
T0
�
save_15/Assign_8Assignpi/dense/kernel/Adamsave_15/RestoreV2:8*"
_class
loc:@pi/dense/kernel*
use_locking(*
_output_shapes

:@*
validate_shape(*
T0
�
save_15/Assign_9Assignpi/dense/kernel/Adam_1save_15/RestoreV2:9*
_output_shapes

:@*
T0*
use_locking(*
validate_shape(*"
_class
loc:@pi/dense/kernel
�
save_15/Assign_10Assignpi/dense_1/biassave_15/RestoreV2:10*
use_locking(*
validate_shape(*
T0*
_output_shapes
:@*"
_class
loc:@pi/dense_1/bias
�
save_15/Assign_11Assignpi/dense_1/bias/Adamsave_15/RestoreV2:11*"
_class
loc:@pi/dense_1/bias*
T0*
_output_shapes
:@*
validate_shape(*
use_locking(
�
save_15/Assign_12Assignpi/dense_1/bias/Adam_1save_15/RestoreV2:12*
use_locking(*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
T0*
_output_shapes
:@
�
save_15/Assign_13Assignpi/dense_1/kernelsave_15/RestoreV2:13*
validate_shape(*
_output_shapes

:@@*$
_class
loc:@pi/dense_1/kernel*
T0*
use_locking(
�
save_15/Assign_14Assignpi/dense_1/kernel/Adamsave_15/RestoreV2:14*
use_locking(*$
_class
loc:@pi/dense_1/kernel*
validate_shape(*
T0*
_output_shapes

:@@
�
save_15/Assign_15Assignpi/dense_1/kernel/Adam_1save_15/RestoreV2:15*
_output_shapes

:@@*$
_class
loc:@pi/dense_1/kernel*
validate_shape(*
T0*
use_locking(
�
save_15/Assign_16Assignpi/dense_2/biassave_15/RestoreV2:16*"
_class
loc:@pi/dense_2/bias*
use_locking(*
validate_shape(*
_output_shapes
:*
T0
�
save_15/Assign_17Assignpi/dense_2/bias/Adamsave_15/RestoreV2:17*
_output_shapes
:*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
T0*
use_locking(
�
save_15/Assign_18Assignpi/dense_2/bias/Adam_1save_15/RestoreV2:18*
T0*
_output_shapes
:*
validate_shape(*
use_locking(*"
_class
loc:@pi/dense_2/bias
�
save_15/Assign_19Assignpi/dense_2/kernelsave_15/RestoreV2:19*
use_locking(*
validate_shape(*
_output_shapes

:@*
T0*$
_class
loc:@pi/dense_2/kernel
�
save_15/Assign_20Assignpi/dense_2/kernel/Adamsave_15/RestoreV2:20*
use_locking(*
T0*
validate_shape(*$
_class
loc:@pi/dense_2/kernel*
_output_shapes

:@
�
save_15/Assign_21Assignpi/dense_2/kernel/Adam_1save_15/RestoreV2:21*
T0*
use_locking(*
_output_shapes

:@*
validate_shape(*$
_class
loc:@pi/dense_2/kernel
�
save_15/Assign_22Assignv/dense/biassave_15/RestoreV2:22*
_class
loc:@v/dense/bias*
use_locking(*
T0*
_output_shapes
:@*
validate_shape(
�
save_15/Assign_23Assignv/dense/bias/Adamsave_15/RestoreV2:23*
T0*
use_locking(*
_class
loc:@v/dense/bias*
_output_shapes
:@*
validate_shape(
�
save_15/Assign_24Assignv/dense/bias/Adam_1save_15/RestoreV2:24*
_output_shapes
:@*
validate_shape(*
use_locking(*
T0*
_class
loc:@v/dense/bias
�
save_15/Assign_25Assignv/dense/kernelsave_15/RestoreV2:25*
T0*!
_class
loc:@v/dense/kernel*
use_locking(*
_output_shapes

:@*
validate_shape(
�
save_15/Assign_26Assignv/dense/kernel/Adamsave_15/RestoreV2:26*
use_locking(*!
_class
loc:@v/dense/kernel*
T0*
validate_shape(*
_output_shapes

:@
�
save_15/Assign_27Assignv/dense/kernel/Adam_1save_15/RestoreV2:27*
validate_shape(*
T0*
use_locking(*
_output_shapes

:@*!
_class
loc:@v/dense/kernel
�
save_15/Assign_28Assignv/dense_1/biassave_15/RestoreV2:28*
T0*
use_locking(*!
_class
loc:@v/dense_1/bias*
_output_shapes
:@*
validate_shape(
�
save_15/Assign_29Assignv/dense_1/bias/Adamsave_15/RestoreV2:29*
_output_shapes
:@*
use_locking(*
validate_shape(*
T0*!
_class
loc:@v/dense_1/bias
�
save_15/Assign_30Assignv/dense_1/bias/Adam_1save_15/RestoreV2:30*
_output_shapes
:@*
use_locking(*
T0*!
_class
loc:@v/dense_1/bias*
validate_shape(
�
save_15/Assign_31Assignv/dense_1/kernelsave_15/RestoreV2:31*#
_class
loc:@v/dense_1/kernel*
T0*
use_locking(*
_output_shapes

:@@*
validate_shape(
�
save_15/Assign_32Assignv/dense_1/kernel/Adamsave_15/RestoreV2:32*
_output_shapes

:@@*#
_class
loc:@v/dense_1/kernel*
use_locking(*
T0*
validate_shape(
�
save_15/Assign_33Assignv/dense_1/kernel/Adam_1save_15/RestoreV2:33*#
_class
loc:@v/dense_1/kernel*
T0*
use_locking(*
_output_shapes

:@@*
validate_shape(
�
save_15/Assign_34Assignv/dense_2/biassave_15/RestoreV2:34*!
_class
loc:@v/dense_2/bias*
T0*
validate_shape(*
_output_shapes
:*
use_locking(
�
save_15/Assign_35Assignv/dense_2/bias/Adamsave_15/RestoreV2:35*!
_class
loc:@v/dense_2/bias*
validate_shape(*
use_locking(*
_output_shapes
:*
T0
�
save_15/Assign_36Assignv/dense_2/bias/Adam_1save_15/RestoreV2:36*
T0*!
_class
loc:@v/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(
�
save_15/Assign_37Assignv/dense_2/kernelsave_15/RestoreV2:37*
T0*#
_class
loc:@v/dense_2/kernel*
use_locking(*
_output_shapes

:@*
validate_shape(
�
save_15/Assign_38Assignv/dense_2/kernel/Adamsave_15/RestoreV2:38*
_output_shapes

:@*
use_locking(*
validate_shape(*#
_class
loc:@v/dense_2/kernel*
T0
�
save_15/Assign_39Assignv/dense_2/kernel/Adam_1save_15/RestoreV2:39*
validate_shape(*
T0*
use_locking(*#
_class
loc:@v/dense_2/kernel*
_output_shapes

:@
�
save_15/restore_shardNoOp^save_15/Assign^save_15/Assign_1^save_15/Assign_10^save_15/Assign_11^save_15/Assign_12^save_15/Assign_13^save_15/Assign_14^save_15/Assign_15^save_15/Assign_16^save_15/Assign_17^save_15/Assign_18^save_15/Assign_19^save_15/Assign_2^save_15/Assign_20^save_15/Assign_21^save_15/Assign_22^save_15/Assign_23^save_15/Assign_24^save_15/Assign_25^save_15/Assign_26^save_15/Assign_27^save_15/Assign_28^save_15/Assign_29^save_15/Assign_3^save_15/Assign_30^save_15/Assign_31^save_15/Assign_32^save_15/Assign_33^save_15/Assign_34^save_15/Assign_35^save_15/Assign_36^save_15/Assign_37^save_15/Assign_38^save_15/Assign_39^save_15/Assign_4^save_15/Assign_5^save_15/Assign_6^save_15/Assign_7^save_15/Assign_8^save_15/Assign_9
3
save_15/restore_allNoOp^save_15/restore_shard
\
save_16/filename/inputConst*
_output_shapes
: *
valueB Bmodel*
dtype0
t
save_16/filenamePlaceholderWithDefaultsave_16/filename/input*
_output_shapes
: *
shape: *
dtype0
k
save_16/ConstPlaceholderWithDefaultsave_16/filename*
dtype0*
shape: *
_output_shapes
: 
�
save_16/StringJoin/inputs_1Const*
dtype0*
_output_shapes
: *<
value3B1 B+_temp_0ff4c433308347cba2b683049300df01/part
~
save_16/StringJoin
StringJoinsave_16/Constsave_16/StringJoin/inputs_1*
_output_shapes
: *
	separator *
N
T
save_16/num_shardsConst*
value	B :*
_output_shapes
: *
dtype0
_
save_16/ShardedFilename/shardConst*
_output_shapes
: *
value	B : *
dtype0
�
save_16/ShardedFilenameShardedFilenamesave_16/StringJoinsave_16/ShardedFilename/shardsave_16/num_shards*
_output_shapes
: 
�
save_16/SaveV2/tensor_namesConst*
_output_shapes
:(*�
value�B�(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
dtype0
�
save_16/SaveV2/shape_and_slicesConst*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:(
�
save_16/SaveV2SaveV2save_16/ShardedFilenamesave_16/SaveV2/tensor_namessave_16/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1v/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1v/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1v/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1v/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1v/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1v/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1*6
dtypes,
*2(
�
save_16/control_dependencyIdentitysave_16/ShardedFilename^save_16/SaveV2*
T0*
_output_shapes
: **
_class 
loc:@save_16/ShardedFilename
�
.save_16/MergeV2Checkpoints/checkpoint_prefixesPacksave_16/ShardedFilename^save_16/control_dependency*
N*

axis *
_output_shapes
:*
T0
�
save_16/MergeV2CheckpointsMergeV2Checkpoints.save_16/MergeV2Checkpoints/checkpoint_prefixessave_16/Const*
delete_old_dirs(
�
save_16/IdentityIdentitysave_16/Const^save_16/MergeV2Checkpoints^save_16/control_dependency*
T0*
_output_shapes
: 
�
save_16/RestoreV2/tensor_namesConst*
dtype0*
_output_shapes
:(*�
value�B�(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1
�
"save_16/RestoreV2/shape_and_slicesConst*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:(*
dtype0
�
save_16/RestoreV2	RestoreV2save_16/Constsave_16/RestoreV2/tensor_names"save_16/RestoreV2/shape_and_slices*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::*6
dtypes,
*2(
�
save_16/AssignAssignbeta1_powersave_16/RestoreV2*
T0*
validate_shape(*
_output_shapes
: * 
_class
loc:@pi/dense/bias*
use_locking(
�
save_16/Assign_1Assignbeta1_power_1save_16/RestoreV2:1*
use_locking(*
_output_shapes
: *
validate_shape(*
_class
loc:@v/dense/bias*
T0
�
save_16/Assign_2Assignbeta2_powersave_16/RestoreV2:2*
T0* 
_class
loc:@pi/dense/bias*
_output_shapes
: *
use_locking(*
validate_shape(
�
save_16/Assign_3Assignbeta2_power_1save_16/RestoreV2:3*
_class
loc:@v/dense/bias*
validate_shape(*
use_locking(*
_output_shapes
: *
T0
�
save_16/Assign_4Assignpi/dense/biassave_16/RestoreV2:4*
_output_shapes
:@*
T0*
validate_shape(*
use_locking(* 
_class
loc:@pi/dense/bias
�
save_16/Assign_5Assignpi/dense/bias/Adamsave_16/RestoreV2:5*
validate_shape(*
_output_shapes
:@*
use_locking(* 
_class
loc:@pi/dense/bias*
T0
�
save_16/Assign_6Assignpi/dense/bias/Adam_1save_16/RestoreV2:6*
_output_shapes
:@*
T0* 
_class
loc:@pi/dense/bias*
use_locking(*
validate_shape(
�
save_16/Assign_7Assignpi/dense/kernelsave_16/RestoreV2:7*
use_locking(*
T0*
_output_shapes

:@*
validate_shape(*"
_class
loc:@pi/dense/kernel
�
save_16/Assign_8Assignpi/dense/kernel/Adamsave_16/RestoreV2:8*
use_locking(*
validate_shape(*
T0*
_output_shapes

:@*"
_class
loc:@pi/dense/kernel
�
save_16/Assign_9Assignpi/dense/kernel/Adam_1save_16/RestoreV2:9*
use_locking(*"
_class
loc:@pi/dense/kernel*
_output_shapes

:@*
validate_shape(*
T0
�
save_16/Assign_10Assignpi/dense_1/biassave_16/RestoreV2:10*
validate_shape(*
use_locking(*
_output_shapes
:@*
T0*"
_class
loc:@pi/dense_1/bias
�
save_16/Assign_11Assignpi/dense_1/bias/Adamsave_16/RestoreV2:11*
T0*
validate_shape(*"
_class
loc:@pi/dense_1/bias*
_output_shapes
:@*
use_locking(
�
save_16/Assign_12Assignpi/dense_1/bias/Adam_1save_16/RestoreV2:12*
T0*
_output_shapes
:@*"
_class
loc:@pi/dense_1/bias*
use_locking(*
validate_shape(
�
save_16/Assign_13Assignpi/dense_1/kernelsave_16/RestoreV2:13*
validate_shape(*
use_locking(*$
_class
loc:@pi/dense_1/kernel*
_output_shapes

:@@*
T0
�
save_16/Assign_14Assignpi/dense_1/kernel/Adamsave_16/RestoreV2:14*
_output_shapes

:@@*
T0*
use_locking(*
validate_shape(*$
_class
loc:@pi/dense_1/kernel
�
save_16/Assign_15Assignpi/dense_1/kernel/Adam_1save_16/RestoreV2:15*$
_class
loc:@pi/dense_1/kernel*
use_locking(*
T0*
_output_shapes

:@@*
validate_shape(
�
save_16/Assign_16Assignpi/dense_2/biassave_16/RestoreV2:16*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:*
validate_shape(*
T0*
use_locking(
�
save_16/Assign_17Assignpi/dense_2/bias/Adamsave_16/RestoreV2:17*
validate_shape(*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:*
T0*
use_locking(
�
save_16/Assign_18Assignpi/dense_2/bias/Adam_1save_16/RestoreV2:18*
T0*
use_locking(*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
_output_shapes
:
�
save_16/Assign_19Assignpi/dense_2/kernelsave_16/RestoreV2:19*
use_locking(*$
_class
loc:@pi/dense_2/kernel*
_output_shapes

:@*
validate_shape(*
T0
�
save_16/Assign_20Assignpi/dense_2/kernel/Adamsave_16/RestoreV2:20*
validate_shape(*
use_locking(*
T0*
_output_shapes

:@*$
_class
loc:@pi/dense_2/kernel
�
save_16/Assign_21Assignpi/dense_2/kernel/Adam_1save_16/RestoreV2:21*$
_class
loc:@pi/dense_2/kernel*
_output_shapes

:@*
validate_shape(*
T0*
use_locking(
�
save_16/Assign_22Assignv/dense/biassave_16/RestoreV2:22*
use_locking(*
_class
loc:@v/dense/bias*
_output_shapes
:@*
T0*
validate_shape(
�
save_16/Assign_23Assignv/dense/bias/Adamsave_16/RestoreV2:23*
use_locking(*
_output_shapes
:@*
T0*
validate_shape(*
_class
loc:@v/dense/bias
�
save_16/Assign_24Assignv/dense/bias/Adam_1save_16/RestoreV2:24*
_output_shapes
:@*
use_locking(*
_class
loc:@v/dense/bias*
validate_shape(*
T0
�
save_16/Assign_25Assignv/dense/kernelsave_16/RestoreV2:25*
use_locking(*
T0*!
_class
loc:@v/dense/kernel*
_output_shapes

:@*
validate_shape(
�
save_16/Assign_26Assignv/dense/kernel/Adamsave_16/RestoreV2:26*
_output_shapes

:@*
validate_shape(*
use_locking(*
T0*!
_class
loc:@v/dense/kernel
�
save_16/Assign_27Assignv/dense/kernel/Adam_1save_16/RestoreV2:27*
use_locking(*!
_class
loc:@v/dense/kernel*
_output_shapes

:@*
validate_shape(*
T0
�
save_16/Assign_28Assignv/dense_1/biassave_16/RestoreV2:28*
_output_shapes
:@*
use_locking(*
validate_shape(*!
_class
loc:@v/dense_1/bias*
T0
�
save_16/Assign_29Assignv/dense_1/bias/Adamsave_16/RestoreV2:29*
validate_shape(*
T0*!
_class
loc:@v/dense_1/bias*
use_locking(*
_output_shapes
:@
�
save_16/Assign_30Assignv/dense_1/bias/Adam_1save_16/RestoreV2:30*!
_class
loc:@v/dense_1/bias*
validate_shape(*
use_locking(*
T0*
_output_shapes
:@
�
save_16/Assign_31Assignv/dense_1/kernelsave_16/RestoreV2:31*
T0*
use_locking(*#
_class
loc:@v/dense_1/kernel*
_output_shapes

:@@*
validate_shape(
�
save_16/Assign_32Assignv/dense_1/kernel/Adamsave_16/RestoreV2:32*
validate_shape(*#
_class
loc:@v/dense_1/kernel*
T0*
_output_shapes

:@@*
use_locking(
�
save_16/Assign_33Assignv/dense_1/kernel/Adam_1save_16/RestoreV2:33*
use_locking(*
_output_shapes

:@@*
validate_shape(*#
_class
loc:@v/dense_1/kernel*
T0
�
save_16/Assign_34Assignv/dense_2/biassave_16/RestoreV2:34*
use_locking(*
_output_shapes
:*
T0*
validate_shape(*!
_class
loc:@v/dense_2/bias
�
save_16/Assign_35Assignv/dense_2/bias/Adamsave_16/RestoreV2:35*
T0*
_output_shapes
:*
validate_shape(*
use_locking(*!
_class
loc:@v/dense_2/bias
�
save_16/Assign_36Assignv/dense_2/bias/Adam_1save_16/RestoreV2:36*
_output_shapes
:*!
_class
loc:@v/dense_2/bias*
T0*
use_locking(*
validate_shape(
�
save_16/Assign_37Assignv/dense_2/kernelsave_16/RestoreV2:37*
use_locking(*
T0*
_output_shapes

:@*
validate_shape(*#
_class
loc:@v/dense_2/kernel
�
save_16/Assign_38Assignv/dense_2/kernel/Adamsave_16/RestoreV2:38*
_output_shapes

:@*
T0*#
_class
loc:@v/dense_2/kernel*
use_locking(*
validate_shape(
�
save_16/Assign_39Assignv/dense_2/kernel/Adam_1save_16/RestoreV2:39*#
_class
loc:@v/dense_2/kernel*
T0*
use_locking(*
validate_shape(*
_output_shapes

:@
�
save_16/restore_shardNoOp^save_16/Assign^save_16/Assign_1^save_16/Assign_10^save_16/Assign_11^save_16/Assign_12^save_16/Assign_13^save_16/Assign_14^save_16/Assign_15^save_16/Assign_16^save_16/Assign_17^save_16/Assign_18^save_16/Assign_19^save_16/Assign_2^save_16/Assign_20^save_16/Assign_21^save_16/Assign_22^save_16/Assign_23^save_16/Assign_24^save_16/Assign_25^save_16/Assign_26^save_16/Assign_27^save_16/Assign_28^save_16/Assign_29^save_16/Assign_3^save_16/Assign_30^save_16/Assign_31^save_16/Assign_32^save_16/Assign_33^save_16/Assign_34^save_16/Assign_35^save_16/Assign_36^save_16/Assign_37^save_16/Assign_38^save_16/Assign_39^save_16/Assign_4^save_16/Assign_5^save_16/Assign_6^save_16/Assign_7^save_16/Assign_8^save_16/Assign_9
3
save_16/restore_allNoOp^save_16/restore_shard
\
save_17/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
t
save_17/filenamePlaceholderWithDefaultsave_17/filename/input*
_output_shapes
: *
shape: *
dtype0
k
save_17/ConstPlaceholderWithDefaultsave_17/filename*
dtype0*
shape: *
_output_shapes
: 
�
save_17/StringJoin/inputs_1Const*
dtype0*
_output_shapes
: *<
value3B1 B+_temp_01e84c690f9848d193544935c811b4bf/part
~
save_17/StringJoin
StringJoinsave_17/Constsave_17/StringJoin/inputs_1*
_output_shapes
: *
N*
	separator 
T
save_17/num_shardsConst*
_output_shapes
: *
value	B :*
dtype0
_
save_17/ShardedFilename/shardConst*
dtype0*
_output_shapes
: *
value	B : 
�
save_17/ShardedFilenameShardedFilenamesave_17/StringJoinsave_17/ShardedFilename/shardsave_17/num_shards*
_output_shapes
: 
�
save_17/SaveV2/tensor_namesConst*
dtype0*�
value�B�(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
_output_shapes
:(
�
save_17/SaveV2/shape_and_slicesConst*
dtype0*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:(
�
save_17/SaveV2SaveV2save_17/ShardedFilenamesave_17/SaveV2/tensor_namessave_17/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1v/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1v/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1v/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1v/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1v/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1v/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1*6
dtypes,
*2(
�
save_17/control_dependencyIdentitysave_17/ShardedFilename^save_17/SaveV2*
_output_shapes
: **
_class 
loc:@save_17/ShardedFilename*
T0
�
.save_17/MergeV2Checkpoints/checkpoint_prefixesPacksave_17/ShardedFilename^save_17/control_dependency*
N*
_output_shapes
:*

axis *
T0
�
save_17/MergeV2CheckpointsMergeV2Checkpoints.save_17/MergeV2Checkpoints/checkpoint_prefixessave_17/Const*
delete_old_dirs(
�
save_17/IdentityIdentitysave_17/Const^save_17/MergeV2Checkpoints^save_17/control_dependency*
T0*
_output_shapes
: 
�
save_17/RestoreV2/tensor_namesConst*
_output_shapes
:(*
dtype0*�
value�B�(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1
�
"save_17/RestoreV2/shape_and_slicesConst*
dtype0*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:(
�
save_17/RestoreV2	RestoreV2save_17/Constsave_17/RestoreV2/tensor_names"save_17/RestoreV2/shape_and_slices*6
dtypes,
*2(*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::
�
save_17/AssignAssignbeta1_powersave_17/RestoreV2*
_output_shapes
: *
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(
�
save_17/Assign_1Assignbeta1_power_1save_17/RestoreV2:1*
_class
loc:@v/dense/bias*
_output_shapes
: *
T0*
validate_shape(*
use_locking(
�
save_17/Assign_2Assignbeta2_powersave_17/RestoreV2:2*
_output_shapes
: *
validate_shape(* 
_class
loc:@pi/dense/bias*
use_locking(*
T0
�
save_17/Assign_3Assignbeta2_power_1save_17/RestoreV2:3*
validate_shape(*
T0*
use_locking(*
_output_shapes
: *
_class
loc:@v/dense/bias
�
save_17/Assign_4Assignpi/dense/biassave_17/RestoreV2:4*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
use_locking(*
_output_shapes
:@
�
save_17/Assign_5Assignpi/dense/bias/Adamsave_17/RestoreV2:5*
use_locking(*
_output_shapes
:@*
validate_shape(*
T0* 
_class
loc:@pi/dense/bias
�
save_17/Assign_6Assignpi/dense/bias/Adam_1save_17/RestoreV2:6*
use_locking(*
T0*
_output_shapes
:@* 
_class
loc:@pi/dense/bias*
validate_shape(
�
save_17/Assign_7Assignpi/dense/kernelsave_17/RestoreV2:7*"
_class
loc:@pi/dense/kernel*
T0*
_output_shapes

:@*
use_locking(*
validate_shape(
�
save_17/Assign_8Assignpi/dense/kernel/Adamsave_17/RestoreV2:8*
use_locking(*
validate_shape(*"
_class
loc:@pi/dense/kernel*
_output_shapes

:@*
T0
�
save_17/Assign_9Assignpi/dense/kernel/Adam_1save_17/RestoreV2:9*
_output_shapes

:@*
use_locking(*"
_class
loc:@pi/dense/kernel*
T0*
validate_shape(
�
save_17/Assign_10Assignpi/dense_1/biassave_17/RestoreV2:10*
validate_shape(*
_output_shapes
:@*
use_locking(*"
_class
loc:@pi/dense_1/bias*
T0
�
save_17/Assign_11Assignpi/dense_1/bias/Adamsave_17/RestoreV2:11*
use_locking(*
_output_shapes
:@*
validate_shape(*
T0*"
_class
loc:@pi/dense_1/bias
�
save_17/Assign_12Assignpi/dense_1/bias/Adam_1save_17/RestoreV2:12*
_output_shapes
:@*
validate_shape(*
use_locking(*"
_class
loc:@pi/dense_1/bias*
T0
�
save_17/Assign_13Assignpi/dense_1/kernelsave_17/RestoreV2:13*
validate_shape(*
T0*
use_locking(*$
_class
loc:@pi/dense_1/kernel*
_output_shapes

:@@
�
save_17/Assign_14Assignpi/dense_1/kernel/Adamsave_17/RestoreV2:14*
T0*
_output_shapes

:@@*
validate_shape(*$
_class
loc:@pi/dense_1/kernel*
use_locking(
�
save_17/Assign_15Assignpi/dense_1/kernel/Adam_1save_17/RestoreV2:15*$
_class
loc:@pi/dense_1/kernel*
T0*
use_locking(*
validate_shape(*
_output_shapes

:@@
�
save_17/Assign_16Assignpi/dense_2/biassave_17/RestoreV2:16*
_output_shapes
:*
validate_shape(*"
_class
loc:@pi/dense_2/bias*
use_locking(*
T0
�
save_17/Assign_17Assignpi/dense_2/bias/Adamsave_17/RestoreV2:17*
T0*
use_locking(*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:*
validate_shape(
�
save_17/Assign_18Assignpi/dense_2/bias/Adam_1save_17/RestoreV2:18*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:*
validate_shape(*
T0*
use_locking(
�
save_17/Assign_19Assignpi/dense_2/kernelsave_17/RestoreV2:19*$
_class
loc:@pi/dense_2/kernel*
use_locking(*
T0*
_output_shapes

:@*
validate_shape(
�
save_17/Assign_20Assignpi/dense_2/kernel/Adamsave_17/RestoreV2:20*
T0*
validate_shape(*$
_class
loc:@pi/dense_2/kernel*
use_locking(*
_output_shapes

:@
�
save_17/Assign_21Assignpi/dense_2/kernel/Adam_1save_17/RestoreV2:21*$
_class
loc:@pi/dense_2/kernel*
use_locking(*
T0*
_output_shapes

:@*
validate_shape(
�
save_17/Assign_22Assignv/dense/biassave_17/RestoreV2:22*
T0*
_class
loc:@v/dense/bias*
_output_shapes
:@*
use_locking(*
validate_shape(
�
save_17/Assign_23Assignv/dense/bias/Adamsave_17/RestoreV2:23*
use_locking(*
_output_shapes
:@*
_class
loc:@v/dense/bias*
T0*
validate_shape(
�
save_17/Assign_24Assignv/dense/bias/Adam_1save_17/RestoreV2:24*
validate_shape(*
T0*
_class
loc:@v/dense/bias*
_output_shapes
:@*
use_locking(
�
save_17/Assign_25Assignv/dense/kernelsave_17/RestoreV2:25*
_output_shapes

:@*
use_locking(*
T0*!
_class
loc:@v/dense/kernel*
validate_shape(
�
save_17/Assign_26Assignv/dense/kernel/Adamsave_17/RestoreV2:26*
T0*
_output_shapes

:@*!
_class
loc:@v/dense/kernel*
use_locking(*
validate_shape(
�
save_17/Assign_27Assignv/dense/kernel/Adam_1save_17/RestoreV2:27*!
_class
loc:@v/dense/kernel*
use_locking(*
_output_shapes

:@*
validate_shape(*
T0
�
save_17/Assign_28Assignv/dense_1/biassave_17/RestoreV2:28*
_output_shapes
:@*
T0*
validate_shape(*!
_class
loc:@v/dense_1/bias*
use_locking(
�
save_17/Assign_29Assignv/dense_1/bias/Adamsave_17/RestoreV2:29*!
_class
loc:@v/dense_1/bias*
use_locking(*
T0*
_output_shapes
:@*
validate_shape(
�
save_17/Assign_30Assignv/dense_1/bias/Adam_1save_17/RestoreV2:30*
use_locking(*!
_class
loc:@v/dense_1/bias*
_output_shapes
:@*
validate_shape(*
T0
�
save_17/Assign_31Assignv/dense_1/kernelsave_17/RestoreV2:31*
use_locking(*
T0*#
_class
loc:@v/dense_1/kernel*
_output_shapes

:@@*
validate_shape(
�
save_17/Assign_32Assignv/dense_1/kernel/Adamsave_17/RestoreV2:32*
validate_shape(*#
_class
loc:@v/dense_1/kernel*
_output_shapes

:@@*
use_locking(*
T0
�
save_17/Assign_33Assignv/dense_1/kernel/Adam_1save_17/RestoreV2:33*
T0*
use_locking(*
_output_shapes

:@@*
validate_shape(*#
_class
loc:@v/dense_1/kernel
�
save_17/Assign_34Assignv/dense_2/biassave_17/RestoreV2:34*!
_class
loc:@v/dense_2/bias*
_output_shapes
:*
T0*
validate_shape(*
use_locking(
�
save_17/Assign_35Assignv/dense_2/bias/Adamsave_17/RestoreV2:35*
use_locking(*
validate_shape(*!
_class
loc:@v/dense_2/bias*
_output_shapes
:*
T0
�
save_17/Assign_36Assignv/dense_2/bias/Adam_1save_17/RestoreV2:36*
_output_shapes
:*
validate_shape(*
T0*
use_locking(*!
_class
loc:@v/dense_2/bias
�
save_17/Assign_37Assignv/dense_2/kernelsave_17/RestoreV2:37*
validate_shape(*
_output_shapes

:@*
use_locking(*#
_class
loc:@v/dense_2/kernel*
T0
�
save_17/Assign_38Assignv/dense_2/kernel/Adamsave_17/RestoreV2:38*
_output_shapes

:@*
validate_shape(*
use_locking(*#
_class
loc:@v/dense_2/kernel*
T0
�
save_17/Assign_39Assignv/dense_2/kernel/Adam_1save_17/RestoreV2:39*
validate_shape(*#
_class
loc:@v/dense_2/kernel*
_output_shapes

:@*
T0*
use_locking(
�
save_17/restore_shardNoOp^save_17/Assign^save_17/Assign_1^save_17/Assign_10^save_17/Assign_11^save_17/Assign_12^save_17/Assign_13^save_17/Assign_14^save_17/Assign_15^save_17/Assign_16^save_17/Assign_17^save_17/Assign_18^save_17/Assign_19^save_17/Assign_2^save_17/Assign_20^save_17/Assign_21^save_17/Assign_22^save_17/Assign_23^save_17/Assign_24^save_17/Assign_25^save_17/Assign_26^save_17/Assign_27^save_17/Assign_28^save_17/Assign_29^save_17/Assign_3^save_17/Assign_30^save_17/Assign_31^save_17/Assign_32^save_17/Assign_33^save_17/Assign_34^save_17/Assign_35^save_17/Assign_36^save_17/Assign_37^save_17/Assign_38^save_17/Assign_39^save_17/Assign_4^save_17/Assign_5^save_17/Assign_6^save_17/Assign_7^save_17/Assign_8^save_17/Assign_9
3
save_17/restore_allNoOp^save_17/restore_shard
\
save_18/filename/inputConst*
dtype0*
_output_shapes
: *
valueB Bmodel
t
save_18/filenamePlaceholderWithDefaultsave_18/filename/input*
dtype0*
shape: *
_output_shapes
: 
k
save_18/ConstPlaceholderWithDefaultsave_18/filename*
dtype0*
shape: *
_output_shapes
: 
�
save_18/StringJoin/inputs_1Const*<
value3B1 B+_temp_9316019320e744c8ab70d19892103ce8/part*
dtype0*
_output_shapes
: 
~
save_18/StringJoin
StringJoinsave_18/Constsave_18/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
T
save_18/num_shardsConst*
_output_shapes
: *
value	B :*
dtype0
_
save_18/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 
�
save_18/ShardedFilenameShardedFilenamesave_18/StringJoinsave_18/ShardedFilename/shardsave_18/num_shards*
_output_shapes
: 
�
save_18/SaveV2/tensor_namesConst*
_output_shapes
:(*�
value�B�(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
dtype0
�
save_18/SaveV2/shape_and_slicesConst*
dtype0*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:(
�
save_18/SaveV2SaveV2save_18/ShardedFilenamesave_18/SaveV2/tensor_namessave_18/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1v/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1v/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1v/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1v/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1v/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1v/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1*6
dtypes,
*2(
�
save_18/control_dependencyIdentitysave_18/ShardedFilename^save_18/SaveV2*
T0**
_class 
loc:@save_18/ShardedFilename*
_output_shapes
: 
�
.save_18/MergeV2Checkpoints/checkpoint_prefixesPacksave_18/ShardedFilename^save_18/control_dependency*
_output_shapes
:*
N*
T0*

axis 
�
save_18/MergeV2CheckpointsMergeV2Checkpoints.save_18/MergeV2Checkpoints/checkpoint_prefixessave_18/Const*
delete_old_dirs(
�
save_18/IdentityIdentitysave_18/Const^save_18/MergeV2Checkpoints^save_18/control_dependency*
_output_shapes
: *
T0
�
save_18/RestoreV2/tensor_namesConst*�
value�B�(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
_output_shapes
:(*
dtype0
�
"save_18/RestoreV2/shape_and_slicesConst*
_output_shapes
:(*
dtype0*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
�
save_18/RestoreV2	RestoreV2save_18/Constsave_18/RestoreV2/tensor_names"save_18/RestoreV2/shape_and_slices*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::*6
dtypes,
*2(
�
save_18/AssignAssignbeta1_powersave_18/RestoreV2* 
_class
loc:@pi/dense/bias*
use_locking(*
_output_shapes
: *
validate_shape(*
T0
�
save_18/Assign_1Assignbeta1_power_1save_18/RestoreV2:1*
validate_shape(*
use_locking(*
_output_shapes
: *
_class
loc:@v/dense/bias*
T0
�
save_18/Assign_2Assignbeta2_powersave_18/RestoreV2:2*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
_output_shapes
: *
validate_shape(
�
save_18/Assign_3Assignbeta2_power_1save_18/RestoreV2:3*
use_locking(*
_class
loc:@v/dense/bias*
validate_shape(*
T0*
_output_shapes
: 
�
save_18/Assign_4Assignpi/dense/biassave_18/RestoreV2:4*
_output_shapes
:@*
use_locking(*
validate_shape(* 
_class
loc:@pi/dense/bias*
T0
�
save_18/Assign_5Assignpi/dense/bias/Adamsave_18/RestoreV2:5*
validate_shape(* 
_class
loc:@pi/dense/bias*
_output_shapes
:@*
use_locking(*
T0
�
save_18/Assign_6Assignpi/dense/bias/Adam_1save_18/RestoreV2:6*
validate_shape(* 
_class
loc:@pi/dense/bias*
_output_shapes
:@*
T0*
use_locking(
�
save_18/Assign_7Assignpi/dense/kernelsave_18/RestoreV2:7*
use_locking(*
_output_shapes

:@*
validate_shape(*"
_class
loc:@pi/dense/kernel*
T0
�
save_18/Assign_8Assignpi/dense/kernel/Adamsave_18/RestoreV2:8*
T0*
use_locking(*
validate_shape(*"
_class
loc:@pi/dense/kernel*
_output_shapes

:@
�
save_18/Assign_9Assignpi/dense/kernel/Adam_1save_18/RestoreV2:9*
use_locking(*
validate_shape(*
_output_shapes

:@*"
_class
loc:@pi/dense/kernel*
T0
�
save_18/Assign_10Assignpi/dense_1/biassave_18/RestoreV2:10*
_output_shapes
:@*
use_locking(*
T0*
validate_shape(*"
_class
loc:@pi/dense_1/bias
�
save_18/Assign_11Assignpi/dense_1/bias/Adamsave_18/RestoreV2:11*"
_class
loc:@pi/dense_1/bias*
use_locking(*
_output_shapes
:@*
T0*
validate_shape(
�
save_18/Assign_12Assignpi/dense_1/bias/Adam_1save_18/RestoreV2:12*
validate_shape(*"
_class
loc:@pi/dense_1/bias*
_output_shapes
:@*
T0*
use_locking(
�
save_18/Assign_13Assignpi/dense_1/kernelsave_18/RestoreV2:13*
_output_shapes

:@@*
T0*
validate_shape(*$
_class
loc:@pi/dense_1/kernel*
use_locking(
�
save_18/Assign_14Assignpi/dense_1/kernel/Adamsave_18/RestoreV2:14*
use_locking(*$
_class
loc:@pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@*
T0
�
save_18/Assign_15Assignpi/dense_1/kernel/Adam_1save_18/RestoreV2:15*
use_locking(*
T0*
_output_shapes

:@@*
validate_shape(*$
_class
loc:@pi/dense_1/kernel
�
save_18/Assign_16Assignpi/dense_2/biassave_18/RestoreV2:16*
_output_shapes
:*
T0*"
_class
loc:@pi/dense_2/bias*
use_locking(*
validate_shape(
�
save_18/Assign_17Assignpi/dense_2/bias/Adamsave_18/RestoreV2:17*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:*
use_locking(*
T0*
validate_shape(
�
save_18/Assign_18Assignpi/dense_2/bias/Adam_1save_18/RestoreV2:18*
T0*
validate_shape(*"
_class
loc:@pi/dense_2/bias*
use_locking(*
_output_shapes
:
�
save_18/Assign_19Assignpi/dense_2/kernelsave_18/RestoreV2:19*
T0*
use_locking(*
_output_shapes

:@*$
_class
loc:@pi/dense_2/kernel*
validate_shape(
�
save_18/Assign_20Assignpi/dense_2/kernel/Adamsave_18/RestoreV2:20*
_output_shapes

:@*$
_class
loc:@pi/dense_2/kernel*
use_locking(*
validate_shape(*
T0
�
save_18/Assign_21Assignpi/dense_2/kernel/Adam_1save_18/RestoreV2:21*$
_class
loc:@pi/dense_2/kernel*
_output_shapes

:@*
use_locking(*
T0*
validate_shape(
�
save_18/Assign_22Assignv/dense/biassave_18/RestoreV2:22*
use_locking(*
validate_shape(*
T0*
_output_shapes
:@*
_class
loc:@v/dense/bias
�
save_18/Assign_23Assignv/dense/bias/Adamsave_18/RestoreV2:23*
_output_shapes
:@*
validate_shape(*
_class
loc:@v/dense/bias*
T0*
use_locking(
�
save_18/Assign_24Assignv/dense/bias/Adam_1save_18/RestoreV2:24*
T0*
validate_shape(*
_output_shapes
:@*
_class
loc:@v/dense/bias*
use_locking(
�
save_18/Assign_25Assignv/dense/kernelsave_18/RestoreV2:25*!
_class
loc:@v/dense/kernel*
T0*
use_locking(*
_output_shapes

:@*
validate_shape(
�
save_18/Assign_26Assignv/dense/kernel/Adamsave_18/RestoreV2:26*!
_class
loc:@v/dense/kernel*
T0*
validate_shape(*
_output_shapes

:@*
use_locking(
�
save_18/Assign_27Assignv/dense/kernel/Adam_1save_18/RestoreV2:27*!
_class
loc:@v/dense/kernel*
validate_shape(*
T0*
use_locking(*
_output_shapes

:@
�
save_18/Assign_28Assignv/dense_1/biassave_18/RestoreV2:28*
use_locking(*
_output_shapes
:@*
T0*!
_class
loc:@v/dense_1/bias*
validate_shape(
�
save_18/Assign_29Assignv/dense_1/bias/Adamsave_18/RestoreV2:29*
use_locking(*!
_class
loc:@v/dense_1/bias*
validate_shape(*
T0*
_output_shapes
:@
�
save_18/Assign_30Assignv/dense_1/bias/Adam_1save_18/RestoreV2:30*
T0*
use_locking(*
validate_shape(*
_output_shapes
:@*!
_class
loc:@v/dense_1/bias
�
save_18/Assign_31Assignv/dense_1/kernelsave_18/RestoreV2:31*
T0*
validate_shape(*
_output_shapes

:@@*#
_class
loc:@v/dense_1/kernel*
use_locking(
�
save_18/Assign_32Assignv/dense_1/kernel/Adamsave_18/RestoreV2:32*
validate_shape(*
T0*
_output_shapes

:@@*#
_class
loc:@v/dense_1/kernel*
use_locking(
�
save_18/Assign_33Assignv/dense_1/kernel/Adam_1save_18/RestoreV2:33*
validate_shape(*
T0*#
_class
loc:@v/dense_1/kernel*
_output_shapes

:@@*
use_locking(
�
save_18/Assign_34Assignv/dense_2/biassave_18/RestoreV2:34*
use_locking(*
_output_shapes
:*
T0*!
_class
loc:@v/dense_2/bias*
validate_shape(
�
save_18/Assign_35Assignv/dense_2/bias/Adamsave_18/RestoreV2:35*
validate_shape(*
T0*!
_class
loc:@v/dense_2/bias*
use_locking(*
_output_shapes
:
�
save_18/Assign_36Assignv/dense_2/bias/Adam_1save_18/RestoreV2:36*
validate_shape(*
use_locking(*!
_class
loc:@v/dense_2/bias*
T0*
_output_shapes
:
�
save_18/Assign_37Assignv/dense_2/kernelsave_18/RestoreV2:37*#
_class
loc:@v/dense_2/kernel*
_output_shapes

:@*
use_locking(*
T0*
validate_shape(
�
save_18/Assign_38Assignv/dense_2/kernel/Adamsave_18/RestoreV2:38*
T0*#
_class
loc:@v/dense_2/kernel*
use_locking(*
_output_shapes

:@*
validate_shape(
�
save_18/Assign_39Assignv/dense_2/kernel/Adam_1save_18/RestoreV2:39*
validate_shape(*
use_locking(*
_output_shapes

:@*
T0*#
_class
loc:@v/dense_2/kernel
�
save_18/restore_shardNoOp^save_18/Assign^save_18/Assign_1^save_18/Assign_10^save_18/Assign_11^save_18/Assign_12^save_18/Assign_13^save_18/Assign_14^save_18/Assign_15^save_18/Assign_16^save_18/Assign_17^save_18/Assign_18^save_18/Assign_19^save_18/Assign_2^save_18/Assign_20^save_18/Assign_21^save_18/Assign_22^save_18/Assign_23^save_18/Assign_24^save_18/Assign_25^save_18/Assign_26^save_18/Assign_27^save_18/Assign_28^save_18/Assign_29^save_18/Assign_3^save_18/Assign_30^save_18/Assign_31^save_18/Assign_32^save_18/Assign_33^save_18/Assign_34^save_18/Assign_35^save_18/Assign_36^save_18/Assign_37^save_18/Assign_38^save_18/Assign_39^save_18/Assign_4^save_18/Assign_5^save_18/Assign_6^save_18/Assign_7^save_18/Assign_8^save_18/Assign_9
3
save_18/restore_allNoOp^save_18/restore_shard
\
save_19/filename/inputConst*
_output_shapes
: *
dtype0*
valueB Bmodel
t
save_19/filenamePlaceholderWithDefaultsave_19/filename/input*
_output_shapes
: *
dtype0*
shape: 
k
save_19/ConstPlaceholderWithDefaultsave_19/filename*
dtype0*
_output_shapes
: *
shape: 
�
save_19/StringJoin/inputs_1Const*
dtype0*
_output_shapes
: *<
value3B1 B+_temp_6c9e0f0a92b8406fb93fe59ef0bade42/part
~
save_19/StringJoin
StringJoinsave_19/Constsave_19/StringJoin/inputs_1*
_output_shapes
: *
	separator *
N
T
save_19/num_shardsConst*
value	B :*
_output_shapes
: *
dtype0
_
save_19/ShardedFilename/shardConst*
dtype0*
_output_shapes
: *
value	B : 
�
save_19/ShardedFilenameShardedFilenamesave_19/StringJoinsave_19/ShardedFilename/shardsave_19/num_shards*
_output_shapes
: 
�
save_19/SaveV2/tensor_namesConst*�
value�B�(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
_output_shapes
:(*
dtype0
�
save_19/SaveV2/shape_and_slicesConst*
dtype0*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:(
�
save_19/SaveV2SaveV2save_19/ShardedFilenamesave_19/SaveV2/tensor_namessave_19/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1v/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1v/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1v/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1v/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1v/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1v/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1*6
dtypes,
*2(
�
save_19/control_dependencyIdentitysave_19/ShardedFilename^save_19/SaveV2*
_output_shapes
: *
T0**
_class 
loc:@save_19/ShardedFilename
�
.save_19/MergeV2Checkpoints/checkpoint_prefixesPacksave_19/ShardedFilename^save_19/control_dependency*
T0*
N*
_output_shapes
:*

axis 
�
save_19/MergeV2CheckpointsMergeV2Checkpoints.save_19/MergeV2Checkpoints/checkpoint_prefixessave_19/Const*
delete_old_dirs(
�
save_19/IdentityIdentitysave_19/Const^save_19/MergeV2Checkpoints^save_19/control_dependency*
_output_shapes
: *
T0
�
save_19/RestoreV2/tensor_namesConst*
dtype0*�
value�B�(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
_output_shapes
:(
�
"save_19/RestoreV2/shape_and_slicesConst*
_output_shapes
:(*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
�
save_19/RestoreV2	RestoreV2save_19/Constsave_19/RestoreV2/tensor_names"save_19/RestoreV2/shape_and_slices*6
dtypes,
*2(*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::
�
save_19/AssignAssignbeta1_powersave_19/RestoreV2*
validate_shape(* 
_class
loc:@pi/dense/bias*
_output_shapes
: *
use_locking(*
T0
�
save_19/Assign_1Assignbeta1_power_1save_19/RestoreV2:1*
_output_shapes
: *
_class
loc:@v/dense/bias*
validate_shape(*
use_locking(*
T0
�
save_19/Assign_2Assignbeta2_powersave_19/RestoreV2:2*
validate_shape(*
use_locking(*
_output_shapes
: *
T0* 
_class
loc:@pi/dense/bias
�
save_19/Assign_3Assignbeta2_power_1save_19/RestoreV2:3*
use_locking(*
validate_shape(*
_output_shapes
: *
_class
loc:@v/dense/bias*
T0
�
save_19/Assign_4Assignpi/dense/biassave_19/RestoreV2:4*
T0*
validate_shape(* 
_class
loc:@pi/dense/bias*
_output_shapes
:@*
use_locking(
�
save_19/Assign_5Assignpi/dense/bias/Adamsave_19/RestoreV2:5*
_output_shapes
:@*
validate_shape(*
T0* 
_class
loc:@pi/dense/bias*
use_locking(
�
save_19/Assign_6Assignpi/dense/bias/Adam_1save_19/RestoreV2:6*
T0*
validate_shape(*
use_locking(* 
_class
loc:@pi/dense/bias*
_output_shapes
:@
�
save_19/Assign_7Assignpi/dense/kernelsave_19/RestoreV2:7*
T0*
_output_shapes

:@*
use_locking(*"
_class
loc:@pi/dense/kernel*
validate_shape(
�
save_19/Assign_8Assignpi/dense/kernel/Adamsave_19/RestoreV2:8*
_output_shapes

:@*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(
�
save_19/Assign_9Assignpi/dense/kernel/Adam_1save_19/RestoreV2:9*
validate_shape(*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel*
_output_shapes

:@
�
save_19/Assign_10Assignpi/dense_1/biassave_19/RestoreV2:10*
_output_shapes
:@*
validate_shape(*
T0*
use_locking(*"
_class
loc:@pi/dense_1/bias
�
save_19/Assign_11Assignpi/dense_1/bias/Adamsave_19/RestoreV2:11*"
_class
loc:@pi/dense_1/bias*
use_locking(*
T0*
validate_shape(*
_output_shapes
:@
�
save_19/Assign_12Assignpi/dense_1/bias/Adam_1save_19/RestoreV2:12*
validate_shape(*
T0*
use_locking(*
_output_shapes
:@*"
_class
loc:@pi/dense_1/bias
�
save_19/Assign_13Assignpi/dense_1/kernelsave_19/RestoreV2:13*
validate_shape(*$
_class
loc:@pi/dense_1/kernel*
T0*
use_locking(*
_output_shapes

:@@
�
save_19/Assign_14Assignpi/dense_1/kernel/Adamsave_19/RestoreV2:14*$
_class
loc:@pi/dense_1/kernel*
_output_shapes

:@@*
validate_shape(*
T0*
use_locking(
�
save_19/Assign_15Assignpi/dense_1/kernel/Adam_1save_19/RestoreV2:15*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel*
_output_shapes

:@@*
validate_shape(
�
save_19/Assign_16Assignpi/dense_2/biassave_19/RestoreV2:16*"
_class
loc:@pi/dense_2/bias*
T0*
validate_shape(*
use_locking(*
_output_shapes
:
�
save_19/Assign_17Assignpi/dense_2/bias/Adamsave_19/RestoreV2:17*
use_locking(*
_output_shapes
:*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
T0
�
save_19/Assign_18Assignpi/dense_2/bias/Adam_1save_19/RestoreV2:18*
_output_shapes
:*
validate_shape(*"
_class
loc:@pi/dense_2/bias*
use_locking(*
T0
�
save_19/Assign_19Assignpi/dense_2/kernelsave_19/RestoreV2:19*
use_locking(*
validate_shape(*$
_class
loc:@pi/dense_2/kernel*
T0*
_output_shapes

:@
�
save_19/Assign_20Assignpi/dense_2/kernel/Adamsave_19/RestoreV2:20*
T0*
use_locking(*
validate_shape(*$
_class
loc:@pi/dense_2/kernel*
_output_shapes

:@
�
save_19/Assign_21Assignpi/dense_2/kernel/Adam_1save_19/RestoreV2:21*
T0*
use_locking(*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@
�
save_19/Assign_22Assignv/dense/biassave_19/RestoreV2:22*
T0*
_output_shapes
:@*
_class
loc:@v/dense/bias*
use_locking(*
validate_shape(
�
save_19/Assign_23Assignv/dense/bias/Adamsave_19/RestoreV2:23*
_output_shapes
:@*
use_locking(*
T0*
_class
loc:@v/dense/bias*
validate_shape(
�
save_19/Assign_24Assignv/dense/bias/Adam_1save_19/RestoreV2:24*
_class
loc:@v/dense/bias*
T0*
_output_shapes
:@*
validate_shape(*
use_locking(
�
save_19/Assign_25Assignv/dense/kernelsave_19/RestoreV2:25*
T0*
_output_shapes

:@*
validate_shape(*!
_class
loc:@v/dense/kernel*
use_locking(
�
save_19/Assign_26Assignv/dense/kernel/Adamsave_19/RestoreV2:26*!
_class
loc:@v/dense/kernel*
use_locking(*
validate_shape(*
_output_shapes

:@*
T0
�
save_19/Assign_27Assignv/dense/kernel/Adam_1save_19/RestoreV2:27*!
_class
loc:@v/dense/kernel*
_output_shapes

:@*
T0*
validate_shape(*
use_locking(
�
save_19/Assign_28Assignv/dense_1/biassave_19/RestoreV2:28*
_output_shapes
:@*
validate_shape(*
use_locking(*
T0*!
_class
loc:@v/dense_1/bias
�
save_19/Assign_29Assignv/dense_1/bias/Adamsave_19/RestoreV2:29*
use_locking(*!
_class
loc:@v/dense_1/bias*
T0*
validate_shape(*
_output_shapes
:@
�
save_19/Assign_30Assignv/dense_1/bias/Adam_1save_19/RestoreV2:30*
_output_shapes
:@*
use_locking(*
validate_shape(*
T0*!
_class
loc:@v/dense_1/bias
�
save_19/Assign_31Assignv/dense_1/kernelsave_19/RestoreV2:31*
validate_shape(*#
_class
loc:@v/dense_1/kernel*
T0*
_output_shapes

:@@*
use_locking(
�
save_19/Assign_32Assignv/dense_1/kernel/Adamsave_19/RestoreV2:32*
T0*#
_class
loc:@v/dense_1/kernel*
use_locking(*
_output_shapes

:@@*
validate_shape(
�
save_19/Assign_33Assignv/dense_1/kernel/Adam_1save_19/RestoreV2:33*
_output_shapes

:@@*
use_locking(*
T0*#
_class
loc:@v/dense_1/kernel*
validate_shape(
�
save_19/Assign_34Assignv/dense_2/biassave_19/RestoreV2:34*
validate_shape(*
_output_shapes
:*!
_class
loc:@v/dense_2/bias*
T0*
use_locking(
�
save_19/Assign_35Assignv/dense_2/bias/Adamsave_19/RestoreV2:35*
_output_shapes
:*
use_locking(*
T0*
validate_shape(*!
_class
loc:@v/dense_2/bias
�
save_19/Assign_36Assignv/dense_2/bias/Adam_1save_19/RestoreV2:36*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*!
_class
loc:@v/dense_2/bias
�
save_19/Assign_37Assignv/dense_2/kernelsave_19/RestoreV2:37*#
_class
loc:@v/dense_2/kernel*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0
�
save_19/Assign_38Assignv/dense_2/kernel/Adamsave_19/RestoreV2:38*#
_class
loc:@v/dense_2/kernel*
_output_shapes

:@*
validate_shape(*
T0*
use_locking(
�
save_19/Assign_39Assignv/dense_2/kernel/Adam_1save_19/RestoreV2:39*
T0*#
_class
loc:@v/dense_2/kernel*
use_locking(*
_output_shapes

:@*
validate_shape(
�
save_19/restore_shardNoOp^save_19/Assign^save_19/Assign_1^save_19/Assign_10^save_19/Assign_11^save_19/Assign_12^save_19/Assign_13^save_19/Assign_14^save_19/Assign_15^save_19/Assign_16^save_19/Assign_17^save_19/Assign_18^save_19/Assign_19^save_19/Assign_2^save_19/Assign_20^save_19/Assign_21^save_19/Assign_22^save_19/Assign_23^save_19/Assign_24^save_19/Assign_25^save_19/Assign_26^save_19/Assign_27^save_19/Assign_28^save_19/Assign_29^save_19/Assign_3^save_19/Assign_30^save_19/Assign_31^save_19/Assign_32^save_19/Assign_33^save_19/Assign_34^save_19/Assign_35^save_19/Assign_36^save_19/Assign_37^save_19/Assign_38^save_19/Assign_39^save_19/Assign_4^save_19/Assign_5^save_19/Assign_6^save_19/Assign_7^save_19/Assign_8^save_19/Assign_9
3
save_19/restore_allNoOp^save_19/restore_shard
\
save_20/filename/inputConst*
dtype0*
_output_shapes
: *
valueB Bmodel
t
save_20/filenamePlaceholderWithDefaultsave_20/filename/input*
dtype0*
_output_shapes
: *
shape: 
k
save_20/ConstPlaceholderWithDefaultsave_20/filename*
_output_shapes
: *
dtype0*
shape: 
�
save_20/StringJoin/inputs_1Const*<
value3B1 B+_temp_2349b617d90c410896627acca04a9c2c/part*
_output_shapes
: *
dtype0
~
save_20/StringJoin
StringJoinsave_20/Constsave_20/StringJoin/inputs_1*
	separator *
_output_shapes
: *
N
T
save_20/num_shardsConst*
_output_shapes
: *
dtype0*
value	B :
_
save_20/ShardedFilename/shardConst*
value	B : *
_output_shapes
: *
dtype0
�
save_20/ShardedFilenameShardedFilenamesave_20/StringJoinsave_20/ShardedFilename/shardsave_20/num_shards*
_output_shapes
: 
�
save_20/SaveV2/tensor_namesConst*
_output_shapes
:(*
dtype0*�
value�B�(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1
�
save_20/SaveV2/shape_and_slicesConst*
_output_shapes
:(*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
�
save_20/SaveV2SaveV2save_20/ShardedFilenamesave_20/SaveV2/tensor_namessave_20/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1v/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1v/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1v/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1v/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1v/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1v/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1*6
dtypes,
*2(
�
save_20/control_dependencyIdentitysave_20/ShardedFilename^save_20/SaveV2*
_output_shapes
: *
T0**
_class 
loc:@save_20/ShardedFilename
�
.save_20/MergeV2Checkpoints/checkpoint_prefixesPacksave_20/ShardedFilename^save_20/control_dependency*
_output_shapes
:*
N*

axis *
T0
�
save_20/MergeV2CheckpointsMergeV2Checkpoints.save_20/MergeV2Checkpoints/checkpoint_prefixessave_20/Const*
delete_old_dirs(
�
save_20/IdentityIdentitysave_20/Const^save_20/MergeV2Checkpoints^save_20/control_dependency*
T0*
_output_shapes
: 
�
save_20/RestoreV2/tensor_namesConst*
dtype0*�
value�B�(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
_output_shapes
:(
�
"save_20/RestoreV2/shape_and_slicesConst*
_output_shapes
:(*
dtype0*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
�
save_20/RestoreV2	RestoreV2save_20/Constsave_20/RestoreV2/tensor_names"save_20/RestoreV2/shape_and_slices*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::*6
dtypes,
*2(
�
save_20/AssignAssignbeta1_powersave_20/RestoreV2*
validate_shape(*
T0* 
_class
loc:@pi/dense/bias*
_output_shapes
: *
use_locking(
�
save_20/Assign_1Assignbeta1_power_1save_20/RestoreV2:1*
validate_shape(*
use_locking(*
_output_shapes
: *
_class
loc:@v/dense/bias*
T0
�
save_20/Assign_2Assignbeta2_powersave_20/RestoreV2:2*
T0* 
_class
loc:@pi/dense/bias*
use_locking(*
_output_shapes
: *
validate_shape(
�
save_20/Assign_3Assignbeta2_power_1save_20/RestoreV2:3*
validate_shape(*
_class
loc:@v/dense/bias*
T0*
_output_shapes
: *
use_locking(
�
save_20/Assign_4Assignpi/dense/biassave_20/RestoreV2:4*
use_locking(*
_output_shapes
:@*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(
�
save_20/Assign_5Assignpi/dense/bias/Adamsave_20/RestoreV2:5*
validate_shape(*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
_output_shapes
:@
�
save_20/Assign_6Assignpi/dense/bias/Adam_1save_20/RestoreV2:6*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
use_locking(*
_output_shapes
:@
�
save_20/Assign_7Assignpi/dense/kernelsave_20/RestoreV2:7*
T0*
validate_shape(*
_output_shapes

:@*
use_locking(*"
_class
loc:@pi/dense/kernel
�
save_20/Assign_8Assignpi/dense/kernel/Adamsave_20/RestoreV2:8*
T0*
validate_shape(*"
_class
loc:@pi/dense/kernel*
use_locking(*
_output_shapes

:@
�
save_20/Assign_9Assignpi/dense/kernel/Adam_1save_20/RestoreV2:9*
use_locking(*
validate_shape(*"
_class
loc:@pi/dense/kernel*
T0*
_output_shapes

:@
�
save_20/Assign_10Assignpi/dense_1/biassave_20/RestoreV2:10*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*"
_class
loc:@pi/dense_1/bias
�
save_20/Assign_11Assignpi/dense_1/bias/Adamsave_20/RestoreV2:11*
use_locking(*"
_class
loc:@pi/dense_1/bias*
_output_shapes
:@*
T0*
validate_shape(
�
save_20/Assign_12Assignpi/dense_1/bias/Adam_1save_20/RestoreV2:12*
validate_shape(*
_output_shapes
:@*"
_class
loc:@pi/dense_1/bias*
T0*
use_locking(
�
save_20/Assign_13Assignpi/dense_1/kernelsave_20/RestoreV2:13*
_output_shapes

:@@*
T0*
validate_shape(*$
_class
loc:@pi/dense_1/kernel*
use_locking(
�
save_20/Assign_14Assignpi/dense_1/kernel/Adamsave_20/RestoreV2:14*
T0*
use_locking(*
_output_shapes

:@@*
validate_shape(*$
_class
loc:@pi/dense_1/kernel
�
save_20/Assign_15Assignpi/dense_1/kernel/Adam_1save_20/RestoreV2:15*
T0*
validate_shape(*$
_class
loc:@pi/dense_1/kernel*
use_locking(*
_output_shapes

:@@
�
save_20/Assign_16Assignpi/dense_2/biassave_20/RestoreV2:16*
_output_shapes
:*
T0*
validate_shape(*
use_locking(*"
_class
loc:@pi/dense_2/bias
�
save_20/Assign_17Assignpi/dense_2/bias/Adamsave_20/RestoreV2:17*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias*
validate_shape(
�
save_20/Assign_18Assignpi/dense_2/bias/Adam_1save_20/RestoreV2:18*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias*
validate_shape(
�
save_20/Assign_19Assignpi/dense_2/kernelsave_20/RestoreV2:19*
_output_shapes

:@*
validate_shape(*
T0*$
_class
loc:@pi/dense_2/kernel*
use_locking(
�
save_20/Assign_20Assignpi/dense_2/kernel/Adamsave_20/RestoreV2:20*
_output_shapes

:@*
validate_shape(*
use_locking(*$
_class
loc:@pi/dense_2/kernel*
T0
�
save_20/Assign_21Assignpi/dense_2/kernel/Adam_1save_20/RestoreV2:21*
_output_shapes

:@*$
_class
loc:@pi/dense_2/kernel*
use_locking(*
validate_shape(*
T0
�
save_20/Assign_22Assignv/dense/biassave_20/RestoreV2:22*
T0*
use_locking(*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
:@
�
save_20/Assign_23Assignv/dense/bias/Adamsave_20/RestoreV2:23*
_class
loc:@v/dense/bias*
_output_shapes
:@*
validate_shape(*
use_locking(*
T0
�
save_20/Assign_24Assignv/dense/bias/Adam_1save_20/RestoreV2:24*
validate_shape(*
_class
loc:@v/dense/bias*
T0*
_output_shapes
:@*
use_locking(
�
save_20/Assign_25Assignv/dense/kernelsave_20/RestoreV2:25*
_output_shapes

:@*!
_class
loc:@v/dense/kernel*
use_locking(*
T0*
validate_shape(
�
save_20/Assign_26Assignv/dense/kernel/Adamsave_20/RestoreV2:26*!
_class
loc:@v/dense/kernel*
T0*
use_locking(*
_output_shapes

:@*
validate_shape(
�
save_20/Assign_27Assignv/dense/kernel/Adam_1save_20/RestoreV2:27*
T0*
use_locking(*
_output_shapes

:@*
validate_shape(*!
_class
loc:@v/dense/kernel
�
save_20/Assign_28Assignv/dense_1/biassave_20/RestoreV2:28*
T0*
validate_shape(*
use_locking(*
_output_shapes
:@*!
_class
loc:@v/dense_1/bias
�
save_20/Assign_29Assignv/dense_1/bias/Adamsave_20/RestoreV2:29*
validate_shape(*
use_locking(*
T0*
_output_shapes
:@*!
_class
loc:@v/dense_1/bias
�
save_20/Assign_30Assignv/dense_1/bias/Adam_1save_20/RestoreV2:30*!
_class
loc:@v/dense_1/bias*
validate_shape(*
use_locking(*
_output_shapes
:@*
T0
�
save_20/Assign_31Assignv/dense_1/kernelsave_20/RestoreV2:31*
_output_shapes

:@@*
validate_shape(*#
_class
loc:@v/dense_1/kernel*
use_locking(*
T0
�
save_20/Assign_32Assignv/dense_1/kernel/Adamsave_20/RestoreV2:32*
T0*
validate_shape(*#
_class
loc:@v/dense_1/kernel*
_output_shapes

:@@*
use_locking(
�
save_20/Assign_33Assignv/dense_1/kernel/Adam_1save_20/RestoreV2:33*
_output_shapes

:@@*
validate_shape(*#
_class
loc:@v/dense_1/kernel*
T0*
use_locking(
�
save_20/Assign_34Assignv/dense_2/biassave_20/RestoreV2:34*
_output_shapes
:*
use_locking(*!
_class
loc:@v/dense_2/bias*
validate_shape(*
T0
�
save_20/Assign_35Assignv/dense_2/bias/Adamsave_20/RestoreV2:35*
use_locking(*
_output_shapes
:*
validate_shape(*!
_class
loc:@v/dense_2/bias*
T0
�
save_20/Assign_36Assignv/dense_2/bias/Adam_1save_20/RestoreV2:36*
validate_shape(*
_output_shapes
:*!
_class
loc:@v/dense_2/bias*
use_locking(*
T0
�
save_20/Assign_37Assignv/dense_2/kernelsave_20/RestoreV2:37*
_output_shapes

:@*
validate_shape(*
T0*#
_class
loc:@v/dense_2/kernel*
use_locking(
�
save_20/Assign_38Assignv/dense_2/kernel/Adamsave_20/RestoreV2:38*#
_class
loc:@v/dense_2/kernel*
use_locking(*
T0*
validate_shape(*
_output_shapes

:@
�
save_20/Assign_39Assignv/dense_2/kernel/Adam_1save_20/RestoreV2:39*#
_class
loc:@v/dense_2/kernel*
use_locking(*
validate_shape(*
T0*
_output_shapes

:@
�
save_20/restore_shardNoOp^save_20/Assign^save_20/Assign_1^save_20/Assign_10^save_20/Assign_11^save_20/Assign_12^save_20/Assign_13^save_20/Assign_14^save_20/Assign_15^save_20/Assign_16^save_20/Assign_17^save_20/Assign_18^save_20/Assign_19^save_20/Assign_2^save_20/Assign_20^save_20/Assign_21^save_20/Assign_22^save_20/Assign_23^save_20/Assign_24^save_20/Assign_25^save_20/Assign_26^save_20/Assign_27^save_20/Assign_28^save_20/Assign_29^save_20/Assign_3^save_20/Assign_30^save_20/Assign_31^save_20/Assign_32^save_20/Assign_33^save_20/Assign_34^save_20/Assign_35^save_20/Assign_36^save_20/Assign_37^save_20/Assign_38^save_20/Assign_39^save_20/Assign_4^save_20/Assign_5^save_20/Assign_6^save_20/Assign_7^save_20/Assign_8^save_20/Assign_9
3
save_20/restore_allNoOp^save_20/restore_shard
\
save_21/filename/inputConst*
valueB Bmodel*
_output_shapes
: *
dtype0
t
save_21/filenamePlaceholderWithDefaultsave_21/filename/input*
dtype0*
_output_shapes
: *
shape: 
k
save_21/ConstPlaceholderWithDefaultsave_21/filename*
shape: *
_output_shapes
: *
dtype0
�
save_21/StringJoin/inputs_1Const*
_output_shapes
: *<
value3B1 B+_temp_2f10c5cd4a8a4d4a9298c0036828535c/part*
dtype0
~
save_21/StringJoin
StringJoinsave_21/Constsave_21/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
T
save_21/num_shardsConst*
value	B :*
_output_shapes
: *
dtype0
_
save_21/ShardedFilename/shardConst*
value	B : *
_output_shapes
: *
dtype0
�
save_21/ShardedFilenameShardedFilenamesave_21/StringJoinsave_21/ShardedFilename/shardsave_21/num_shards*
_output_shapes
: 
�
save_21/SaveV2/tensor_namesConst*�
value�B�(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:(
�
save_21/SaveV2/shape_and_slicesConst*
_output_shapes
:(*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
�
save_21/SaveV2SaveV2save_21/ShardedFilenamesave_21/SaveV2/tensor_namessave_21/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1v/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1v/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1v/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1v/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1v/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1v/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1*6
dtypes,
*2(
�
save_21/control_dependencyIdentitysave_21/ShardedFilename^save_21/SaveV2*
T0**
_class 
loc:@save_21/ShardedFilename*
_output_shapes
: 
�
.save_21/MergeV2Checkpoints/checkpoint_prefixesPacksave_21/ShardedFilename^save_21/control_dependency*
T0*
N*
_output_shapes
:*

axis 
�
save_21/MergeV2CheckpointsMergeV2Checkpoints.save_21/MergeV2Checkpoints/checkpoint_prefixessave_21/Const*
delete_old_dirs(
�
save_21/IdentityIdentitysave_21/Const^save_21/MergeV2Checkpoints^save_21/control_dependency*
_output_shapes
: *
T0
�
save_21/RestoreV2/tensor_namesConst*
dtype0*
_output_shapes
:(*�
value�B�(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1
�
"save_21/RestoreV2/shape_and_slicesConst*
dtype0*
_output_shapes
:(*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
�
save_21/RestoreV2	RestoreV2save_21/Constsave_21/RestoreV2/tensor_names"save_21/RestoreV2/shape_and_slices*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::*6
dtypes,
*2(
�
save_21/AssignAssignbeta1_powersave_21/RestoreV2*
use_locking(* 
_class
loc:@pi/dense/bias*
_output_shapes
: *
T0*
validate_shape(
�
save_21/Assign_1Assignbeta1_power_1save_21/RestoreV2:1*
validate_shape(*
use_locking(*
_output_shapes
: *
T0*
_class
loc:@v/dense/bias
�
save_21/Assign_2Assignbeta2_powersave_21/RestoreV2:2*
validate_shape(* 
_class
loc:@pi/dense/bias*
T0*
use_locking(*
_output_shapes
: 
�
save_21/Assign_3Assignbeta2_power_1save_21/RestoreV2:3*
validate_shape(*
_output_shapes
: *
T0*
use_locking(*
_class
loc:@v/dense/bias
�
save_21/Assign_4Assignpi/dense/biassave_21/RestoreV2:4*
use_locking(*
T0*
_output_shapes
:@* 
_class
loc:@pi/dense/bias*
validate_shape(
�
save_21/Assign_5Assignpi/dense/bias/Adamsave_21/RestoreV2:5*
validate_shape(*
_output_shapes
:@* 
_class
loc:@pi/dense/bias*
use_locking(*
T0
�
save_21/Assign_6Assignpi/dense/bias/Adam_1save_21/RestoreV2:6*
validate_shape(*
T0* 
_class
loc:@pi/dense/bias*
_output_shapes
:@*
use_locking(
�
save_21/Assign_7Assignpi/dense/kernelsave_21/RestoreV2:7*
use_locking(*"
_class
loc:@pi/dense/kernel*
T0*
_output_shapes

:@*
validate_shape(
�
save_21/Assign_8Assignpi/dense/kernel/Adamsave_21/RestoreV2:8*"
_class
loc:@pi/dense/kernel*
T0*
use_locking(*
validate_shape(*
_output_shapes

:@
�
save_21/Assign_9Assignpi/dense/kernel/Adam_1save_21/RestoreV2:9*
use_locking(*
validate_shape(*
_output_shapes

:@*"
_class
loc:@pi/dense/kernel*
T0
�
save_21/Assign_10Assignpi/dense_1/biassave_21/RestoreV2:10*
validate_shape(*"
_class
loc:@pi/dense_1/bias*
T0*
_output_shapes
:@*
use_locking(
�
save_21/Assign_11Assignpi/dense_1/bias/Adamsave_21/RestoreV2:11*"
_class
loc:@pi/dense_1/bias*
use_locking(*
validate_shape(*
T0*
_output_shapes
:@
�
save_21/Assign_12Assignpi/dense_1/bias/Adam_1save_21/RestoreV2:12*
T0*
validate_shape(*
use_locking(*"
_class
loc:@pi/dense_1/bias*
_output_shapes
:@
�
save_21/Assign_13Assignpi/dense_1/kernelsave_21/RestoreV2:13*
validate_shape(*
_output_shapes

:@@*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel
�
save_21/Assign_14Assignpi/dense_1/kernel/Adamsave_21/RestoreV2:14*
validate_shape(*
_output_shapes

:@@*$
_class
loc:@pi/dense_1/kernel*
use_locking(*
T0
�
save_21/Assign_15Assignpi/dense_1/kernel/Adam_1save_21/RestoreV2:15*$
_class
loc:@pi/dense_1/kernel*
T0*
use_locking(*
validate_shape(*
_output_shapes

:@@
�
save_21/Assign_16Assignpi/dense_2/biassave_21/RestoreV2:16*"
_class
loc:@pi/dense_2/bias*
use_locking(*
_output_shapes
:*
validate_shape(*
T0
�
save_21/Assign_17Assignpi/dense_2/bias/Adamsave_21/RestoreV2:17*
T0*
use_locking(*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:*
validate_shape(
�
save_21/Assign_18Assignpi/dense_2/bias/Adam_1save_21/RestoreV2:18*
use_locking(*
_output_shapes
:*
T0*"
_class
loc:@pi/dense_2/bias*
validate_shape(
�
save_21/Assign_19Assignpi/dense_2/kernelsave_21/RestoreV2:19*
_output_shapes

:@*
use_locking(*
T0*
validate_shape(*$
_class
loc:@pi/dense_2/kernel
�
save_21/Assign_20Assignpi/dense_2/kernel/Adamsave_21/RestoreV2:20*
use_locking(*
validate_shape(*
T0*$
_class
loc:@pi/dense_2/kernel*
_output_shapes

:@
�
save_21/Assign_21Assignpi/dense_2/kernel/Adam_1save_21/RestoreV2:21*
_output_shapes

:@*
use_locking(*
validate_shape(*
T0*$
_class
loc:@pi/dense_2/kernel
�
save_21/Assign_22Assignv/dense/biassave_21/RestoreV2:22*
_class
loc:@v/dense/bias*
validate_shape(*
T0*
use_locking(*
_output_shapes
:@
�
save_21/Assign_23Assignv/dense/bias/Adamsave_21/RestoreV2:23*
use_locking(*
_class
loc:@v/dense/bias*
_output_shapes
:@*
T0*
validate_shape(
�
save_21/Assign_24Assignv/dense/bias/Adam_1save_21/RestoreV2:24*
validate_shape(*
T0*
_output_shapes
:@*
_class
loc:@v/dense/bias*
use_locking(
�
save_21/Assign_25Assignv/dense/kernelsave_21/RestoreV2:25*
use_locking(*
validate_shape(*!
_class
loc:@v/dense/kernel*
T0*
_output_shapes

:@
�
save_21/Assign_26Assignv/dense/kernel/Adamsave_21/RestoreV2:26*
use_locking(*
validate_shape(*
T0*
_output_shapes

:@*!
_class
loc:@v/dense/kernel
�
save_21/Assign_27Assignv/dense/kernel/Adam_1save_21/RestoreV2:27*
T0*
_output_shapes

:@*!
_class
loc:@v/dense/kernel*
validate_shape(*
use_locking(
�
save_21/Assign_28Assignv/dense_1/biassave_21/RestoreV2:28*!
_class
loc:@v/dense_1/bias*
T0*
use_locking(*
validate_shape(*
_output_shapes
:@
�
save_21/Assign_29Assignv/dense_1/bias/Adamsave_21/RestoreV2:29*
T0*
use_locking(*!
_class
loc:@v/dense_1/bias*
validate_shape(*
_output_shapes
:@
�
save_21/Assign_30Assignv/dense_1/bias/Adam_1save_21/RestoreV2:30*
use_locking(*!
_class
loc:@v/dense_1/bias*
_output_shapes
:@*
validate_shape(*
T0
�
save_21/Assign_31Assignv/dense_1/kernelsave_21/RestoreV2:31*
_output_shapes

:@@*
use_locking(*
T0*#
_class
loc:@v/dense_1/kernel*
validate_shape(
�
save_21/Assign_32Assignv/dense_1/kernel/Adamsave_21/RestoreV2:32*
_output_shapes

:@@*
use_locking(*
T0*#
_class
loc:@v/dense_1/kernel*
validate_shape(
�
save_21/Assign_33Assignv/dense_1/kernel/Adam_1save_21/RestoreV2:33*
T0*
validate_shape(*
_output_shapes

:@@*#
_class
loc:@v/dense_1/kernel*
use_locking(
�
save_21/Assign_34Assignv/dense_2/biassave_21/RestoreV2:34*!
_class
loc:@v/dense_2/bias*
use_locking(*
validate_shape(*
T0*
_output_shapes
:
�
save_21/Assign_35Assignv/dense_2/bias/Adamsave_21/RestoreV2:35*
validate_shape(*
_output_shapes
:*!
_class
loc:@v/dense_2/bias*
use_locking(*
T0
�
save_21/Assign_36Assignv/dense_2/bias/Adam_1save_21/RestoreV2:36*
use_locking(*
validate_shape(*
T0*!
_class
loc:@v/dense_2/bias*
_output_shapes
:
�
save_21/Assign_37Assignv/dense_2/kernelsave_21/RestoreV2:37*
T0*#
_class
loc:@v/dense_2/kernel*
_output_shapes

:@*
use_locking(*
validate_shape(
�
save_21/Assign_38Assignv/dense_2/kernel/Adamsave_21/RestoreV2:38*
_output_shapes

:@*
validate_shape(*#
_class
loc:@v/dense_2/kernel*
T0*
use_locking(
�
save_21/Assign_39Assignv/dense_2/kernel/Adam_1save_21/RestoreV2:39*
validate_shape(*
T0*#
_class
loc:@v/dense_2/kernel*
_output_shapes

:@*
use_locking(
�
save_21/restore_shardNoOp^save_21/Assign^save_21/Assign_1^save_21/Assign_10^save_21/Assign_11^save_21/Assign_12^save_21/Assign_13^save_21/Assign_14^save_21/Assign_15^save_21/Assign_16^save_21/Assign_17^save_21/Assign_18^save_21/Assign_19^save_21/Assign_2^save_21/Assign_20^save_21/Assign_21^save_21/Assign_22^save_21/Assign_23^save_21/Assign_24^save_21/Assign_25^save_21/Assign_26^save_21/Assign_27^save_21/Assign_28^save_21/Assign_29^save_21/Assign_3^save_21/Assign_30^save_21/Assign_31^save_21/Assign_32^save_21/Assign_33^save_21/Assign_34^save_21/Assign_35^save_21/Assign_36^save_21/Assign_37^save_21/Assign_38^save_21/Assign_39^save_21/Assign_4^save_21/Assign_5^save_21/Assign_6^save_21/Assign_7^save_21/Assign_8^save_21/Assign_9
3
save_21/restore_allNoOp^save_21/restore_shard
\
save_22/filename/inputConst*
_output_shapes
: *
dtype0*
valueB Bmodel
t
save_22/filenamePlaceholderWithDefaultsave_22/filename/input*
shape: *
_output_shapes
: *
dtype0
k
save_22/ConstPlaceholderWithDefaultsave_22/filename*
_output_shapes
: *
shape: *
dtype0
�
save_22/StringJoin/inputs_1Const*<
value3B1 B+_temp_bb211375f1f44c4bbc0292cc0de29332/part*
dtype0*
_output_shapes
: 
~
save_22/StringJoin
StringJoinsave_22/Constsave_22/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
T
save_22/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
_
save_22/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 
�
save_22/ShardedFilenameShardedFilenamesave_22/StringJoinsave_22/ShardedFilename/shardsave_22/num_shards*
_output_shapes
: 
�
save_22/SaveV2/tensor_namesConst*
_output_shapes
:(*�
value�B�(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
dtype0
�
save_22/SaveV2/shape_and_slicesConst*
_output_shapes
:(*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
�
save_22/SaveV2SaveV2save_22/ShardedFilenamesave_22/SaveV2/tensor_namessave_22/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1v/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1v/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1v/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1v/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1v/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1v/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1*6
dtypes,
*2(
�
save_22/control_dependencyIdentitysave_22/ShardedFilename^save_22/SaveV2*
T0*
_output_shapes
: **
_class 
loc:@save_22/ShardedFilename
�
.save_22/MergeV2Checkpoints/checkpoint_prefixesPacksave_22/ShardedFilename^save_22/control_dependency*
N*

axis *
T0*
_output_shapes
:
�
save_22/MergeV2CheckpointsMergeV2Checkpoints.save_22/MergeV2Checkpoints/checkpoint_prefixessave_22/Const*
delete_old_dirs(
�
save_22/IdentityIdentitysave_22/Const^save_22/MergeV2Checkpoints^save_22/control_dependency*
_output_shapes
: *
T0
�
save_22/RestoreV2/tensor_namesConst*
dtype0*�
value�B�(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
_output_shapes
:(
�
"save_22/RestoreV2/shape_and_slicesConst*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:(
�
save_22/RestoreV2	RestoreV2save_22/Constsave_22/RestoreV2/tensor_names"save_22/RestoreV2/shape_and_slices*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::*6
dtypes,
*2(
�
save_22/AssignAssignbeta1_powersave_22/RestoreV2*
validate_shape(*
_output_shapes
: *
use_locking(*
T0* 
_class
loc:@pi/dense/bias
�
save_22/Assign_1Assignbeta1_power_1save_22/RestoreV2:1*
use_locking(*
T0*
validate_shape(*
_output_shapes
: *
_class
loc:@v/dense/bias
�
save_22/Assign_2Assignbeta2_powersave_22/RestoreV2:2* 
_class
loc:@pi/dense/bias*
_output_shapes
: *
use_locking(*
T0*
validate_shape(
�
save_22/Assign_3Assignbeta2_power_1save_22/RestoreV2:3*
T0*
use_locking(*
_output_shapes
: *
validate_shape(*
_class
loc:@v/dense/bias
�
save_22/Assign_4Assignpi/dense/biassave_22/RestoreV2:4*
_output_shapes
:@* 
_class
loc:@pi/dense/bias*
T0*
use_locking(*
validate_shape(
�
save_22/Assign_5Assignpi/dense/bias/Adamsave_22/RestoreV2:5*
_output_shapes
:@*
T0*
use_locking(* 
_class
loc:@pi/dense/bias*
validate_shape(
�
save_22/Assign_6Assignpi/dense/bias/Adam_1save_22/RestoreV2:6* 
_class
loc:@pi/dense/bias*
_output_shapes
:@*
T0*
use_locking(*
validate_shape(
�
save_22/Assign_7Assignpi/dense/kernelsave_22/RestoreV2:7*
validate_shape(*
use_locking(*"
_class
loc:@pi/dense/kernel*
T0*
_output_shapes

:@
�
save_22/Assign_8Assignpi/dense/kernel/Adamsave_22/RestoreV2:8*
use_locking(*
validate_shape(*
_output_shapes

:@*"
_class
loc:@pi/dense/kernel*
T0
�
save_22/Assign_9Assignpi/dense/kernel/Adam_1save_22/RestoreV2:9*
_output_shapes

:@*
validate_shape(*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel
�
save_22/Assign_10Assignpi/dense_1/biassave_22/RestoreV2:10*
validate_shape(*
_output_shapes
:@*"
_class
loc:@pi/dense_1/bias*
T0*
use_locking(
�
save_22/Assign_11Assignpi/dense_1/bias/Adamsave_22/RestoreV2:11*
T0*"
_class
loc:@pi/dense_1/bias*
use_locking(*
_output_shapes
:@*
validate_shape(
�
save_22/Assign_12Assignpi/dense_1/bias/Adam_1save_22/RestoreV2:12*
use_locking(*
T0*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
_output_shapes
:@
�
save_22/Assign_13Assignpi/dense_1/kernelsave_22/RestoreV2:13*
_output_shapes

:@@*
validate_shape(*$
_class
loc:@pi/dense_1/kernel*
use_locking(*
T0
�
save_22/Assign_14Assignpi/dense_1/kernel/Adamsave_22/RestoreV2:14*
use_locking(*
_output_shapes

:@@*$
_class
loc:@pi/dense_1/kernel*
T0*
validate_shape(
�
save_22/Assign_15Assignpi/dense_1/kernel/Adam_1save_22/RestoreV2:15*
_output_shapes

:@@*$
_class
loc:@pi/dense_1/kernel*
validate_shape(*
use_locking(*
T0
�
save_22/Assign_16Assignpi/dense_2/biassave_22/RestoreV2:16*
use_locking(*
validate_shape(*
T0*
_output_shapes
:*"
_class
loc:@pi/dense_2/bias
�
save_22/Assign_17Assignpi/dense_2/bias/Adamsave_22/RestoreV2:17*
use_locking(*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
T0*
_output_shapes
:
�
save_22/Assign_18Assignpi/dense_2/bias/Adam_1save_22/RestoreV2:18*
validate_shape(*
T0*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:*
use_locking(
�
save_22/Assign_19Assignpi/dense_2/kernelsave_22/RestoreV2:19*
use_locking(*
validate_shape(*
T0*$
_class
loc:@pi/dense_2/kernel*
_output_shapes

:@
�
save_22/Assign_20Assignpi/dense_2/kernel/Adamsave_22/RestoreV2:20*$
_class
loc:@pi/dense_2/kernel*
_output_shapes

:@*
validate_shape(*
T0*
use_locking(
�
save_22/Assign_21Assignpi/dense_2/kernel/Adam_1save_22/RestoreV2:21*
_output_shapes

:@*$
_class
loc:@pi/dense_2/kernel*
T0*
validate_shape(*
use_locking(
�
save_22/Assign_22Assignv/dense/biassave_22/RestoreV2:22*
validate_shape(*
use_locking(*
T0*
_output_shapes
:@*
_class
loc:@v/dense/bias
�
save_22/Assign_23Assignv/dense/bias/Adamsave_22/RestoreV2:23*
validate_shape(*
T0*
_class
loc:@v/dense/bias*
use_locking(*
_output_shapes
:@
�
save_22/Assign_24Assignv/dense/bias/Adam_1save_22/RestoreV2:24*
use_locking(*
T0*
_output_shapes
:@*
_class
loc:@v/dense/bias*
validate_shape(
�
save_22/Assign_25Assignv/dense/kernelsave_22/RestoreV2:25*
T0*!
_class
loc:@v/dense/kernel*
_output_shapes

:@*
validate_shape(*
use_locking(
�
save_22/Assign_26Assignv/dense/kernel/Adamsave_22/RestoreV2:26*
use_locking(*
_output_shapes

:@*
T0*
validate_shape(*!
_class
loc:@v/dense/kernel
�
save_22/Assign_27Assignv/dense/kernel/Adam_1save_22/RestoreV2:27*!
_class
loc:@v/dense/kernel*
validate_shape(*
T0*
use_locking(*
_output_shapes

:@
�
save_22/Assign_28Assignv/dense_1/biassave_22/RestoreV2:28*
use_locking(*
_output_shapes
:@*
validate_shape(*!
_class
loc:@v/dense_1/bias*
T0
�
save_22/Assign_29Assignv/dense_1/bias/Adamsave_22/RestoreV2:29*!
_class
loc:@v/dense_1/bias*
_output_shapes
:@*
validate_shape(*
use_locking(*
T0
�
save_22/Assign_30Assignv/dense_1/bias/Adam_1save_22/RestoreV2:30*
_output_shapes
:@*
T0*
validate_shape(*
use_locking(*!
_class
loc:@v/dense_1/bias
�
save_22/Assign_31Assignv/dense_1/kernelsave_22/RestoreV2:31*
use_locking(*
T0*
_output_shapes

:@@*#
_class
loc:@v/dense_1/kernel*
validate_shape(
�
save_22/Assign_32Assignv/dense_1/kernel/Adamsave_22/RestoreV2:32*
validate_shape(*#
_class
loc:@v/dense_1/kernel*
T0*
_output_shapes

:@@*
use_locking(
�
save_22/Assign_33Assignv/dense_1/kernel/Adam_1save_22/RestoreV2:33*
T0*#
_class
loc:@v/dense_1/kernel*
validate_shape(*
use_locking(*
_output_shapes

:@@
�
save_22/Assign_34Assignv/dense_2/biassave_22/RestoreV2:34*
_output_shapes
:*!
_class
loc:@v/dense_2/bias*
use_locking(*
T0*
validate_shape(
�
save_22/Assign_35Assignv/dense_2/bias/Adamsave_22/RestoreV2:35*
_output_shapes
:*!
_class
loc:@v/dense_2/bias*
use_locking(*
validate_shape(*
T0
�
save_22/Assign_36Assignv/dense_2/bias/Adam_1save_22/RestoreV2:36*
T0*
validate_shape(*
_output_shapes
:*!
_class
loc:@v/dense_2/bias*
use_locking(
�
save_22/Assign_37Assignv/dense_2/kernelsave_22/RestoreV2:37*
validate_shape(*
T0*
_output_shapes

:@*#
_class
loc:@v/dense_2/kernel*
use_locking(
�
save_22/Assign_38Assignv/dense_2/kernel/Adamsave_22/RestoreV2:38*
T0*
_output_shapes

:@*#
_class
loc:@v/dense_2/kernel*
validate_shape(*
use_locking(
�
save_22/Assign_39Assignv/dense_2/kernel/Adam_1save_22/RestoreV2:39*
T0*
_output_shapes

:@*#
_class
loc:@v/dense_2/kernel*
use_locking(*
validate_shape(
�
save_22/restore_shardNoOp^save_22/Assign^save_22/Assign_1^save_22/Assign_10^save_22/Assign_11^save_22/Assign_12^save_22/Assign_13^save_22/Assign_14^save_22/Assign_15^save_22/Assign_16^save_22/Assign_17^save_22/Assign_18^save_22/Assign_19^save_22/Assign_2^save_22/Assign_20^save_22/Assign_21^save_22/Assign_22^save_22/Assign_23^save_22/Assign_24^save_22/Assign_25^save_22/Assign_26^save_22/Assign_27^save_22/Assign_28^save_22/Assign_29^save_22/Assign_3^save_22/Assign_30^save_22/Assign_31^save_22/Assign_32^save_22/Assign_33^save_22/Assign_34^save_22/Assign_35^save_22/Assign_36^save_22/Assign_37^save_22/Assign_38^save_22/Assign_39^save_22/Assign_4^save_22/Assign_5^save_22/Assign_6^save_22/Assign_7^save_22/Assign_8^save_22/Assign_9
3
save_22/restore_allNoOp^save_22/restore_shard
\
save_23/filename/inputConst*
_output_shapes
: *
dtype0*
valueB Bmodel
t
save_23/filenamePlaceholderWithDefaultsave_23/filename/input*
shape: *
_output_shapes
: *
dtype0
k
save_23/ConstPlaceholderWithDefaultsave_23/filename*
dtype0*
_output_shapes
: *
shape: 
�
save_23/StringJoin/inputs_1Const*<
value3B1 B+_temp_9709db3247654303bd1978b7b86adb70/part*
dtype0*
_output_shapes
: 
~
save_23/StringJoin
StringJoinsave_23/Constsave_23/StringJoin/inputs_1*
_output_shapes
: *
N*
	separator 
T
save_23/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
_
save_23/ShardedFilename/shardConst*
_output_shapes
: *
value	B : *
dtype0
�
save_23/ShardedFilenameShardedFilenamesave_23/StringJoinsave_23/ShardedFilename/shardsave_23/num_shards*
_output_shapes
: 
�
save_23/SaveV2/tensor_namesConst*�
value�B�(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:(
�
save_23/SaveV2/shape_and_slicesConst*
dtype0*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:(
�
save_23/SaveV2SaveV2save_23/ShardedFilenamesave_23/SaveV2/tensor_namessave_23/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1v/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1v/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1v/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1v/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1v/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1v/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1*6
dtypes,
*2(
�
save_23/control_dependencyIdentitysave_23/ShardedFilename^save_23/SaveV2*
T0*
_output_shapes
: **
_class 
loc:@save_23/ShardedFilename
�
.save_23/MergeV2Checkpoints/checkpoint_prefixesPacksave_23/ShardedFilename^save_23/control_dependency*
_output_shapes
:*
N*

axis *
T0
�
save_23/MergeV2CheckpointsMergeV2Checkpoints.save_23/MergeV2Checkpoints/checkpoint_prefixessave_23/Const*
delete_old_dirs(
�
save_23/IdentityIdentitysave_23/Const^save_23/MergeV2Checkpoints^save_23/control_dependency*
_output_shapes
: *
T0
�
save_23/RestoreV2/tensor_namesConst*
dtype0*�
value�B�(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
_output_shapes
:(
�
"save_23/RestoreV2/shape_and_slicesConst*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:(*
dtype0
�
save_23/RestoreV2	RestoreV2save_23/Constsave_23/RestoreV2/tensor_names"save_23/RestoreV2/shape_and_slices*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::*6
dtypes,
*2(
�
save_23/AssignAssignbeta1_powersave_23/RestoreV2*
validate_shape(*
_output_shapes
: *
use_locking(* 
_class
loc:@pi/dense/bias*
T0
�
save_23/Assign_1Assignbeta1_power_1save_23/RestoreV2:1*
use_locking(*
_output_shapes
: *
validate_shape(*
_class
loc:@v/dense/bias*
T0
�
save_23/Assign_2Assignbeta2_powersave_23/RestoreV2:2*
validate_shape(*
_output_shapes
: *
T0* 
_class
loc:@pi/dense/bias*
use_locking(
�
save_23/Assign_3Assignbeta2_power_1save_23/RestoreV2:3*
use_locking(*
_output_shapes
: *
T0*
_class
loc:@v/dense/bias*
validate_shape(
�
save_23/Assign_4Assignpi/dense/biassave_23/RestoreV2:4*
use_locking(*
validate_shape(* 
_class
loc:@pi/dense/bias*
T0*
_output_shapes
:@
�
save_23/Assign_5Assignpi/dense/bias/Adamsave_23/RestoreV2:5*
validate_shape(*
_output_shapes
:@*
use_locking(* 
_class
loc:@pi/dense/bias*
T0
�
save_23/Assign_6Assignpi/dense/bias/Adam_1save_23/RestoreV2:6*
_output_shapes
:@* 
_class
loc:@pi/dense/bias*
use_locking(*
validate_shape(*
T0
�
save_23/Assign_7Assignpi/dense/kernelsave_23/RestoreV2:7*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes

:@
�
save_23/Assign_8Assignpi/dense/kernel/Adamsave_23/RestoreV2:8*
T0*
use_locking(*
validate_shape(*"
_class
loc:@pi/dense/kernel*
_output_shapes

:@
�
save_23/Assign_9Assignpi/dense/kernel/Adam_1save_23/RestoreV2:9*
use_locking(*"
_class
loc:@pi/dense/kernel*
T0*
_output_shapes

:@*
validate_shape(
�
save_23/Assign_10Assignpi/dense_1/biassave_23/RestoreV2:10*"
_class
loc:@pi/dense_1/bias*
_output_shapes
:@*
validate_shape(*
T0*
use_locking(
�
save_23/Assign_11Assignpi/dense_1/bias/Adamsave_23/RestoreV2:11*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
use_locking(*
T0*
_output_shapes
:@
�
save_23/Assign_12Assignpi/dense_1/bias/Adam_1save_23/RestoreV2:12*
validate_shape(*
_output_shapes
:@*"
_class
loc:@pi/dense_1/bias*
T0*
use_locking(
�
save_23/Assign_13Assignpi/dense_1/kernelsave_23/RestoreV2:13*$
_class
loc:@pi/dense_1/kernel*
T0*
_output_shapes

:@@*
use_locking(*
validate_shape(
�
save_23/Assign_14Assignpi/dense_1/kernel/Adamsave_23/RestoreV2:14*
T0*
use_locking(*$
_class
loc:@pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
�
save_23/Assign_15Assignpi/dense_1/kernel/Adam_1save_23/RestoreV2:15*
_output_shapes

:@@*
use_locking(*
validate_shape(*$
_class
loc:@pi/dense_1/kernel*
T0
�
save_23/Assign_16Assignpi/dense_2/biassave_23/RestoreV2:16*
T0*"
_class
loc:@pi/dense_2/bias*
use_locking(*
_output_shapes
:*
validate_shape(
�
save_23/Assign_17Assignpi/dense_2/bias/Adamsave_23/RestoreV2:17*
_output_shapes
:*
use_locking(*"
_class
loc:@pi/dense_2/bias*
T0*
validate_shape(
�
save_23/Assign_18Assignpi/dense_2/bias/Adam_1save_23/RestoreV2:18*
validate_shape(*
use_locking(*
_output_shapes
:*"
_class
loc:@pi/dense_2/bias*
T0
�
save_23/Assign_19Assignpi/dense_2/kernelsave_23/RestoreV2:19*$
_class
loc:@pi/dense_2/kernel*
_output_shapes

:@*
use_locking(*
validate_shape(*
T0
�
save_23/Assign_20Assignpi/dense_2/kernel/Adamsave_23/RestoreV2:20*
_output_shapes

:@*
validate_shape(*$
_class
loc:@pi/dense_2/kernel*
T0*
use_locking(
�
save_23/Assign_21Assignpi/dense_2/kernel/Adam_1save_23/RestoreV2:21*
validate_shape(*$
_class
loc:@pi/dense_2/kernel*
use_locking(*
T0*
_output_shapes

:@
�
save_23/Assign_22Assignv/dense/biassave_23/RestoreV2:22*
use_locking(*
T0*
validate_shape(*
_output_shapes
:@*
_class
loc:@v/dense/bias
�
save_23/Assign_23Assignv/dense/bias/Adamsave_23/RestoreV2:23*
_class
loc:@v/dense/bias*
validate_shape(*
use_locking(*
T0*
_output_shapes
:@
�
save_23/Assign_24Assignv/dense/bias/Adam_1save_23/RestoreV2:24*
use_locking(*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
:@
�
save_23/Assign_25Assignv/dense/kernelsave_23/RestoreV2:25*
_output_shapes

:@*
use_locking(*
T0*!
_class
loc:@v/dense/kernel*
validate_shape(
�
save_23/Assign_26Assignv/dense/kernel/Adamsave_23/RestoreV2:26*
T0*!
_class
loc:@v/dense/kernel*
use_locking(*
_output_shapes

:@*
validate_shape(
�
save_23/Assign_27Assignv/dense/kernel/Adam_1save_23/RestoreV2:27*
_output_shapes

:@*
use_locking(*
T0*
validate_shape(*!
_class
loc:@v/dense/kernel
�
save_23/Assign_28Assignv/dense_1/biassave_23/RestoreV2:28*
validate_shape(*
use_locking(*
_output_shapes
:@*
T0*!
_class
loc:@v/dense_1/bias
�
save_23/Assign_29Assignv/dense_1/bias/Adamsave_23/RestoreV2:29*!
_class
loc:@v/dense_1/bias*
_output_shapes
:@*
T0*
use_locking(*
validate_shape(
�
save_23/Assign_30Assignv/dense_1/bias/Adam_1save_23/RestoreV2:30*
T0*
validate_shape(*
use_locking(*
_output_shapes
:@*!
_class
loc:@v/dense_1/bias
�
save_23/Assign_31Assignv/dense_1/kernelsave_23/RestoreV2:31*
_output_shapes

:@@*
use_locking(*
validate_shape(*#
_class
loc:@v/dense_1/kernel*
T0
�
save_23/Assign_32Assignv/dense_1/kernel/Adamsave_23/RestoreV2:32*
validate_shape(*
_output_shapes

:@@*
T0*#
_class
loc:@v/dense_1/kernel*
use_locking(
�
save_23/Assign_33Assignv/dense_1/kernel/Adam_1save_23/RestoreV2:33*
use_locking(*
_output_shapes

:@@*
validate_shape(*
T0*#
_class
loc:@v/dense_1/kernel
�
save_23/Assign_34Assignv/dense_2/biassave_23/RestoreV2:34*!
_class
loc:@v/dense_2/bias*
validate_shape(*
use_locking(*
T0*
_output_shapes
:
�
save_23/Assign_35Assignv/dense_2/bias/Adamsave_23/RestoreV2:35*
validate_shape(*
use_locking(*!
_class
loc:@v/dense_2/bias*
T0*
_output_shapes
:
�
save_23/Assign_36Assignv/dense_2/bias/Adam_1save_23/RestoreV2:36*
_output_shapes
:*
T0*
use_locking(*
validate_shape(*!
_class
loc:@v/dense_2/bias
�
save_23/Assign_37Assignv/dense_2/kernelsave_23/RestoreV2:37*
T0*
validate_shape(*
use_locking(*
_output_shapes

:@*#
_class
loc:@v/dense_2/kernel
�
save_23/Assign_38Assignv/dense_2/kernel/Adamsave_23/RestoreV2:38*
_output_shapes

:@*
validate_shape(*#
_class
loc:@v/dense_2/kernel*
T0*
use_locking(
�
save_23/Assign_39Assignv/dense_2/kernel/Adam_1save_23/RestoreV2:39*#
_class
loc:@v/dense_2/kernel*
_output_shapes

:@*
validate_shape(*
use_locking(*
T0
�
save_23/restore_shardNoOp^save_23/Assign^save_23/Assign_1^save_23/Assign_10^save_23/Assign_11^save_23/Assign_12^save_23/Assign_13^save_23/Assign_14^save_23/Assign_15^save_23/Assign_16^save_23/Assign_17^save_23/Assign_18^save_23/Assign_19^save_23/Assign_2^save_23/Assign_20^save_23/Assign_21^save_23/Assign_22^save_23/Assign_23^save_23/Assign_24^save_23/Assign_25^save_23/Assign_26^save_23/Assign_27^save_23/Assign_28^save_23/Assign_29^save_23/Assign_3^save_23/Assign_30^save_23/Assign_31^save_23/Assign_32^save_23/Assign_33^save_23/Assign_34^save_23/Assign_35^save_23/Assign_36^save_23/Assign_37^save_23/Assign_38^save_23/Assign_39^save_23/Assign_4^save_23/Assign_5^save_23/Assign_6^save_23/Assign_7^save_23/Assign_8^save_23/Assign_9
3
save_23/restore_allNoOp^save_23/restore_shard
\
save_24/filename/inputConst*
dtype0*
valueB Bmodel*
_output_shapes
: 
t
save_24/filenamePlaceholderWithDefaultsave_24/filename/input*
shape: *
dtype0*
_output_shapes
: 
k
save_24/ConstPlaceholderWithDefaultsave_24/filename*
dtype0*
_output_shapes
: *
shape: 
�
save_24/StringJoin/inputs_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+_temp_2145144ea50b4a3aaf327defe9459be9/part
~
save_24/StringJoin
StringJoinsave_24/Constsave_24/StringJoin/inputs_1*
_output_shapes
: *
N*
	separator 
T
save_24/num_shardsConst*
_output_shapes
: *
value	B :*
dtype0
_
save_24/ShardedFilename/shardConst*
dtype0*
value	B : *
_output_shapes
: 
�
save_24/ShardedFilenameShardedFilenamesave_24/StringJoinsave_24/ShardedFilename/shardsave_24/num_shards*
_output_shapes
: 
�
save_24/SaveV2/tensor_namesConst*
dtype0*�
value�B�(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
_output_shapes
:(
�
save_24/SaveV2/shape_and_slicesConst*
dtype0*
_output_shapes
:(*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
�
save_24/SaveV2SaveV2save_24/ShardedFilenamesave_24/SaveV2/tensor_namessave_24/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1v/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1v/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1v/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1v/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1v/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1v/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1*6
dtypes,
*2(
�
save_24/control_dependencyIdentitysave_24/ShardedFilename^save_24/SaveV2**
_class 
loc:@save_24/ShardedFilename*
_output_shapes
: *
T0
�
.save_24/MergeV2Checkpoints/checkpoint_prefixesPacksave_24/ShardedFilename^save_24/control_dependency*
T0*

axis *
N*
_output_shapes
:
�
save_24/MergeV2CheckpointsMergeV2Checkpoints.save_24/MergeV2Checkpoints/checkpoint_prefixessave_24/Const*
delete_old_dirs(
�
save_24/IdentityIdentitysave_24/Const^save_24/MergeV2Checkpoints^save_24/control_dependency*
T0*
_output_shapes
: 
�
save_24/RestoreV2/tensor_namesConst*
_output_shapes
:(*
dtype0*�
value�B�(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1
�
"save_24/RestoreV2/shape_and_slicesConst*
_output_shapes
:(*
dtype0*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
�
save_24/RestoreV2	RestoreV2save_24/Constsave_24/RestoreV2/tensor_names"save_24/RestoreV2/shape_and_slices*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::*6
dtypes,
*2(
�
save_24/AssignAssignbeta1_powersave_24/RestoreV2*
use_locking(*
_output_shapes
: *
T0*
validate_shape(* 
_class
loc:@pi/dense/bias
�
save_24/Assign_1Assignbeta1_power_1save_24/RestoreV2:1*
_output_shapes
: *
T0*
use_locking(*
_class
loc:@v/dense/bias*
validate_shape(
�
save_24/Assign_2Assignbeta2_powersave_24/RestoreV2:2*
validate_shape(*
use_locking(*
_output_shapes
: *
T0* 
_class
loc:@pi/dense/bias
�
save_24/Assign_3Assignbeta2_power_1save_24/RestoreV2:3*
_class
loc:@v/dense/bias*
T0*
_output_shapes
: *
validate_shape(*
use_locking(
�
save_24/Assign_4Assignpi/dense/biassave_24/RestoreV2:4* 
_class
loc:@pi/dense/bias*
T0*
validate_shape(*
use_locking(*
_output_shapes
:@
�
save_24/Assign_5Assignpi/dense/bias/Adamsave_24/RestoreV2:5*
use_locking(*
_output_shapes
:@*
validate_shape(* 
_class
loc:@pi/dense/bias*
T0
�
save_24/Assign_6Assignpi/dense/bias/Adam_1save_24/RestoreV2:6*
use_locking(* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
:@*
T0
�
save_24/Assign_7Assignpi/dense/kernelsave_24/RestoreV2:7*
validate_shape(*
T0*"
_class
loc:@pi/dense/kernel*
use_locking(*
_output_shapes

:@
�
save_24/Assign_8Assignpi/dense/kernel/Adamsave_24/RestoreV2:8*
validate_shape(*"
_class
loc:@pi/dense/kernel*
use_locking(*
T0*
_output_shapes

:@
�
save_24/Assign_9Assignpi/dense/kernel/Adam_1save_24/RestoreV2:9*
T0*
_output_shapes

:@*"
_class
loc:@pi/dense/kernel*
use_locking(*
validate_shape(
�
save_24/Assign_10Assignpi/dense_1/biassave_24/RestoreV2:10*
T0*
use_locking(*
validate_shape(*
_output_shapes
:@*"
_class
loc:@pi/dense_1/bias
�
save_24/Assign_11Assignpi/dense_1/bias/Adamsave_24/RestoreV2:11*
T0*
validate_shape(*
_output_shapes
:@*
use_locking(*"
_class
loc:@pi/dense_1/bias
�
save_24/Assign_12Assignpi/dense_1/bias/Adam_1save_24/RestoreV2:12*
use_locking(*"
_class
loc:@pi/dense_1/bias*
_output_shapes
:@*
T0*
validate_shape(
�
save_24/Assign_13Assignpi/dense_1/kernelsave_24/RestoreV2:13*
validate_shape(*$
_class
loc:@pi/dense_1/kernel*
use_locking(*
_output_shapes

:@@*
T0
�
save_24/Assign_14Assignpi/dense_1/kernel/Adamsave_24/RestoreV2:14*
T0*
use_locking(*$
_class
loc:@pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
�
save_24/Assign_15Assignpi/dense_1/kernel/Adam_1save_24/RestoreV2:15*
use_locking(*
validate_shape(*
T0*$
_class
loc:@pi/dense_1/kernel*
_output_shapes

:@@
�
save_24/Assign_16Assignpi/dense_2/biassave_24/RestoreV2:16*
T0*
use_locking(*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:*
validate_shape(
�
save_24/Assign_17Assignpi/dense_2/bias/Adamsave_24/RestoreV2:17*
validate_shape(*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:*
use_locking(*
T0
�
save_24/Assign_18Assignpi/dense_2/bias/Adam_1save_24/RestoreV2:18*"
_class
loc:@pi/dense_2/bias*
T0*
_output_shapes
:*
use_locking(*
validate_shape(
�
save_24/Assign_19Assignpi/dense_2/kernelsave_24/RestoreV2:19*
use_locking(*$
_class
loc:@pi/dense_2/kernel*
T0*
validate_shape(*
_output_shapes

:@
�
save_24/Assign_20Assignpi/dense_2/kernel/Adamsave_24/RestoreV2:20*
_output_shapes

:@*
use_locking(*
T0*
validate_shape(*$
_class
loc:@pi/dense_2/kernel
�
save_24/Assign_21Assignpi/dense_2/kernel/Adam_1save_24/RestoreV2:21*
_output_shapes

:@*$
_class
loc:@pi/dense_2/kernel*
use_locking(*
T0*
validate_shape(
�
save_24/Assign_22Assignv/dense/biassave_24/RestoreV2:22*
_class
loc:@v/dense/bias*
_output_shapes
:@*
T0*
use_locking(*
validate_shape(
�
save_24/Assign_23Assignv/dense/bias/Adamsave_24/RestoreV2:23*
use_locking(*
validate_shape(*
_output_shapes
:@*
_class
loc:@v/dense/bias*
T0
�
save_24/Assign_24Assignv/dense/bias/Adam_1save_24/RestoreV2:24*
T0*
_output_shapes
:@*
validate_shape(*
use_locking(*
_class
loc:@v/dense/bias
�
save_24/Assign_25Assignv/dense/kernelsave_24/RestoreV2:25*
use_locking(*!
_class
loc:@v/dense/kernel*
validate_shape(*
T0*
_output_shapes

:@
�
save_24/Assign_26Assignv/dense/kernel/Adamsave_24/RestoreV2:26*
_output_shapes

:@*!
_class
loc:@v/dense/kernel*
T0*
use_locking(*
validate_shape(
�
save_24/Assign_27Assignv/dense/kernel/Adam_1save_24/RestoreV2:27*
_output_shapes

:@*!
_class
loc:@v/dense/kernel*
validate_shape(*
T0*
use_locking(
�
save_24/Assign_28Assignv/dense_1/biassave_24/RestoreV2:28*!
_class
loc:@v/dense_1/bias*
T0*
_output_shapes
:@*
use_locking(*
validate_shape(
�
save_24/Assign_29Assignv/dense_1/bias/Adamsave_24/RestoreV2:29*!
_class
loc:@v/dense_1/bias*
validate_shape(*
T0*
use_locking(*
_output_shapes
:@
�
save_24/Assign_30Assignv/dense_1/bias/Adam_1save_24/RestoreV2:30*
T0*
use_locking(*
validate_shape(*
_output_shapes
:@*!
_class
loc:@v/dense_1/bias
�
save_24/Assign_31Assignv/dense_1/kernelsave_24/RestoreV2:31*
use_locking(*
T0*
_output_shapes

:@@*#
_class
loc:@v/dense_1/kernel*
validate_shape(
�
save_24/Assign_32Assignv/dense_1/kernel/Adamsave_24/RestoreV2:32*#
_class
loc:@v/dense_1/kernel*
validate_shape(*
T0*
_output_shapes

:@@*
use_locking(
�
save_24/Assign_33Assignv/dense_1/kernel/Adam_1save_24/RestoreV2:33*
validate_shape(*
use_locking(*
T0*
_output_shapes

:@@*#
_class
loc:@v/dense_1/kernel
�
save_24/Assign_34Assignv/dense_2/biassave_24/RestoreV2:34*
use_locking(*!
_class
loc:@v/dense_2/bias*
_output_shapes
:*
validate_shape(*
T0
�
save_24/Assign_35Assignv/dense_2/bias/Adamsave_24/RestoreV2:35*!
_class
loc:@v/dense_2/bias*
use_locking(*
_output_shapes
:*
validate_shape(*
T0
�
save_24/Assign_36Assignv/dense_2/bias/Adam_1save_24/RestoreV2:36*
T0*
use_locking(*
_output_shapes
:*!
_class
loc:@v/dense_2/bias*
validate_shape(
�
save_24/Assign_37Assignv/dense_2/kernelsave_24/RestoreV2:37*#
_class
loc:@v/dense_2/kernel*
_output_shapes

:@*
use_locking(*
validate_shape(*
T0
�
save_24/Assign_38Assignv/dense_2/kernel/Adamsave_24/RestoreV2:38*
validate_shape(*
_output_shapes

:@*
T0*#
_class
loc:@v/dense_2/kernel*
use_locking(
�
save_24/Assign_39Assignv/dense_2/kernel/Adam_1save_24/RestoreV2:39*
T0*
_output_shapes

:@*#
_class
loc:@v/dense_2/kernel*
validate_shape(*
use_locking(
�
save_24/restore_shardNoOp^save_24/Assign^save_24/Assign_1^save_24/Assign_10^save_24/Assign_11^save_24/Assign_12^save_24/Assign_13^save_24/Assign_14^save_24/Assign_15^save_24/Assign_16^save_24/Assign_17^save_24/Assign_18^save_24/Assign_19^save_24/Assign_2^save_24/Assign_20^save_24/Assign_21^save_24/Assign_22^save_24/Assign_23^save_24/Assign_24^save_24/Assign_25^save_24/Assign_26^save_24/Assign_27^save_24/Assign_28^save_24/Assign_29^save_24/Assign_3^save_24/Assign_30^save_24/Assign_31^save_24/Assign_32^save_24/Assign_33^save_24/Assign_34^save_24/Assign_35^save_24/Assign_36^save_24/Assign_37^save_24/Assign_38^save_24/Assign_39^save_24/Assign_4^save_24/Assign_5^save_24/Assign_6^save_24/Assign_7^save_24/Assign_8^save_24/Assign_9
3
save_24/restore_allNoOp^save_24/restore_shard
\
save_25/filename/inputConst*
_output_shapes
: *
dtype0*
valueB Bmodel
t
save_25/filenamePlaceholderWithDefaultsave_25/filename/input*
shape: *
dtype0*
_output_shapes
: 
k
save_25/ConstPlaceholderWithDefaultsave_25/filename*
dtype0*
_output_shapes
: *
shape: 
�
save_25/StringJoin/inputs_1Const*
dtype0*
_output_shapes
: *<
value3B1 B+_temp_e66e6dc9bc2f4aa0b4ca44fa4bd7969c/part
~
save_25/StringJoin
StringJoinsave_25/Constsave_25/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
T
save_25/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
_
save_25/ShardedFilename/shardConst*
dtype0*
_output_shapes
: *
value	B : 
�
save_25/ShardedFilenameShardedFilenamesave_25/StringJoinsave_25/ShardedFilename/shardsave_25/num_shards*
_output_shapes
: 
�
save_25/SaveV2/tensor_namesConst*�
value�B�(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
_output_shapes
:(*
dtype0
�
save_25/SaveV2/shape_and_slicesConst*
_output_shapes
:(*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
�
save_25/SaveV2SaveV2save_25/ShardedFilenamesave_25/SaveV2/tensor_namessave_25/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1v/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1v/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1v/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1v/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1v/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1v/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1*6
dtypes,
*2(
�
save_25/control_dependencyIdentitysave_25/ShardedFilename^save_25/SaveV2**
_class 
loc:@save_25/ShardedFilename*
T0*
_output_shapes
: 
�
.save_25/MergeV2Checkpoints/checkpoint_prefixesPacksave_25/ShardedFilename^save_25/control_dependency*
N*
T0*
_output_shapes
:*

axis 
�
save_25/MergeV2CheckpointsMergeV2Checkpoints.save_25/MergeV2Checkpoints/checkpoint_prefixessave_25/Const*
delete_old_dirs(
�
save_25/IdentityIdentitysave_25/Const^save_25/MergeV2Checkpoints^save_25/control_dependency*
T0*
_output_shapes
: 
�
save_25/RestoreV2/tensor_namesConst*�
value�B�(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:(
�
"save_25/RestoreV2/shape_and_slicesConst*
_output_shapes
:(*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
�
save_25/RestoreV2	RestoreV2save_25/Constsave_25/RestoreV2/tensor_names"save_25/RestoreV2/shape_and_slices*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::*6
dtypes,
*2(
�
save_25/AssignAssignbeta1_powersave_25/RestoreV2*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
use_locking(*
_output_shapes
: 
�
save_25/Assign_1Assignbeta1_power_1save_25/RestoreV2:1*
use_locking(*
_class
loc:@v/dense/bias*
_output_shapes
: *
T0*
validate_shape(
�
save_25/Assign_2Assignbeta2_powersave_25/RestoreV2:2*
validate_shape(*
use_locking(* 
_class
loc:@pi/dense/bias*
_output_shapes
: *
T0
�
save_25/Assign_3Assignbeta2_power_1save_25/RestoreV2:3*
_class
loc:@v/dense/bias*
T0*
use_locking(*
_output_shapes
: *
validate_shape(
�
save_25/Assign_4Assignpi/dense/biassave_25/RestoreV2:4*
_output_shapes
:@*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
use_locking(
�
save_25/Assign_5Assignpi/dense/bias/Adamsave_25/RestoreV2:5* 
_class
loc:@pi/dense/bias*
T0*
validate_shape(*
_output_shapes
:@*
use_locking(
�
save_25/Assign_6Assignpi/dense/bias/Adam_1save_25/RestoreV2:6*
_output_shapes
:@*
use_locking(*
T0*
validate_shape(* 
_class
loc:@pi/dense/bias
�
save_25/Assign_7Assignpi/dense/kernelsave_25/RestoreV2:7*
_output_shapes

:@*"
_class
loc:@pi/dense/kernel*
validate_shape(*
T0*
use_locking(
�
save_25/Assign_8Assignpi/dense/kernel/Adamsave_25/RestoreV2:8*"
_class
loc:@pi/dense/kernel*
T0*
_output_shapes

:@*
validate_shape(*
use_locking(
�
save_25/Assign_9Assignpi/dense/kernel/Adam_1save_25/RestoreV2:9*
use_locking(*
validate_shape(*
_output_shapes

:@*
T0*"
_class
loc:@pi/dense/kernel
�
save_25/Assign_10Assignpi/dense_1/biassave_25/RestoreV2:10*
_output_shapes
:@*
validate_shape(*
T0*
use_locking(*"
_class
loc:@pi/dense_1/bias
�
save_25/Assign_11Assignpi/dense_1/bias/Adamsave_25/RestoreV2:11*
use_locking(*
validate_shape(*
_output_shapes
:@*
T0*"
_class
loc:@pi/dense_1/bias
�
save_25/Assign_12Assignpi/dense_1/bias/Adam_1save_25/RestoreV2:12*
use_locking(*
validate_shape(*
T0*
_output_shapes
:@*"
_class
loc:@pi/dense_1/bias
�
save_25/Assign_13Assignpi/dense_1/kernelsave_25/RestoreV2:13*$
_class
loc:@pi/dense_1/kernel*
validate_shape(*
use_locking(*
T0*
_output_shapes

:@@
�
save_25/Assign_14Assignpi/dense_1/kernel/Adamsave_25/RestoreV2:14*
validate_shape(*
T0*
use_locking(*$
_class
loc:@pi/dense_1/kernel*
_output_shapes

:@@
�
save_25/Assign_15Assignpi/dense_1/kernel/Adam_1save_25/RestoreV2:15*
validate_shape(*
_output_shapes

:@@*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel
�
save_25/Assign_16Assignpi/dense_2/biassave_25/RestoreV2:16*
T0*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:*
validate_shape(*
use_locking(
�
save_25/Assign_17Assignpi/dense_2/bias/Adamsave_25/RestoreV2:17*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
_output_shapes
:
�
save_25/Assign_18Assignpi/dense_2/bias/Adam_1save_25/RestoreV2:18*
validate_shape(*
use_locking(*
T0*
_output_shapes
:*"
_class
loc:@pi/dense_2/bias
�
save_25/Assign_19Assignpi/dense_2/kernelsave_25/RestoreV2:19*
T0*
_output_shapes

:@*
use_locking(*
validate_shape(*$
_class
loc:@pi/dense_2/kernel
�
save_25/Assign_20Assignpi/dense_2/kernel/Adamsave_25/RestoreV2:20*
_output_shapes

:@*$
_class
loc:@pi/dense_2/kernel*
use_locking(*
T0*
validate_shape(
�
save_25/Assign_21Assignpi/dense_2/kernel/Adam_1save_25/RestoreV2:21*
use_locking(*$
_class
loc:@pi/dense_2/kernel*
T0*
_output_shapes

:@*
validate_shape(
�
save_25/Assign_22Assignv/dense/biassave_25/RestoreV2:22*
_output_shapes
:@*
T0*
validate_shape(*
use_locking(*
_class
loc:@v/dense/bias
�
save_25/Assign_23Assignv/dense/bias/Adamsave_25/RestoreV2:23*
use_locking(*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
:@
�
save_25/Assign_24Assignv/dense/bias/Adam_1save_25/RestoreV2:24*
use_locking(*
validate_shape(*
_output_shapes
:@*
_class
loc:@v/dense/bias*
T0
�
save_25/Assign_25Assignv/dense/kernelsave_25/RestoreV2:25*!
_class
loc:@v/dense/kernel*
_output_shapes

:@*
use_locking(*
T0*
validate_shape(
�
save_25/Assign_26Assignv/dense/kernel/Adamsave_25/RestoreV2:26*
validate_shape(*
use_locking(*
T0*
_output_shapes

:@*!
_class
loc:@v/dense/kernel
�
save_25/Assign_27Assignv/dense/kernel/Adam_1save_25/RestoreV2:27*
_output_shapes

:@*
use_locking(*
validate_shape(*
T0*!
_class
loc:@v/dense/kernel
�
save_25/Assign_28Assignv/dense_1/biassave_25/RestoreV2:28*
T0*!
_class
loc:@v/dense_1/bias*
use_locking(*
_output_shapes
:@*
validate_shape(
�
save_25/Assign_29Assignv/dense_1/bias/Adamsave_25/RestoreV2:29*
T0*!
_class
loc:@v/dense_1/bias*
validate_shape(*
use_locking(*
_output_shapes
:@
�
save_25/Assign_30Assignv/dense_1/bias/Adam_1save_25/RestoreV2:30*!
_class
loc:@v/dense_1/bias*
_output_shapes
:@*
validate_shape(*
use_locking(*
T0
�
save_25/Assign_31Assignv/dense_1/kernelsave_25/RestoreV2:31*
_output_shapes

:@@*#
_class
loc:@v/dense_1/kernel*
use_locking(*
validate_shape(*
T0
�
save_25/Assign_32Assignv/dense_1/kernel/Adamsave_25/RestoreV2:32*#
_class
loc:@v/dense_1/kernel*
T0*
_output_shapes

:@@*
use_locking(*
validate_shape(
�
save_25/Assign_33Assignv/dense_1/kernel/Adam_1save_25/RestoreV2:33*
validate_shape(*
use_locking(*
_output_shapes

:@@*#
_class
loc:@v/dense_1/kernel*
T0
�
save_25/Assign_34Assignv/dense_2/biassave_25/RestoreV2:34*
validate_shape(*
use_locking(*!
_class
loc:@v/dense_2/bias*
_output_shapes
:*
T0
�
save_25/Assign_35Assignv/dense_2/bias/Adamsave_25/RestoreV2:35*
validate_shape(*
T0*
_output_shapes
:*
use_locking(*!
_class
loc:@v/dense_2/bias
�
save_25/Assign_36Assignv/dense_2/bias/Adam_1save_25/RestoreV2:36*
_output_shapes
:*
use_locking(*!
_class
loc:@v/dense_2/bias*
T0*
validate_shape(
�
save_25/Assign_37Assignv/dense_2/kernelsave_25/RestoreV2:37*
T0*
validate_shape(*
_output_shapes

:@*
use_locking(*#
_class
loc:@v/dense_2/kernel
�
save_25/Assign_38Assignv/dense_2/kernel/Adamsave_25/RestoreV2:38*
use_locking(*#
_class
loc:@v/dense_2/kernel*
validate_shape(*
T0*
_output_shapes

:@
�
save_25/Assign_39Assignv/dense_2/kernel/Adam_1save_25/RestoreV2:39*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0*#
_class
loc:@v/dense_2/kernel
�
save_25/restore_shardNoOp^save_25/Assign^save_25/Assign_1^save_25/Assign_10^save_25/Assign_11^save_25/Assign_12^save_25/Assign_13^save_25/Assign_14^save_25/Assign_15^save_25/Assign_16^save_25/Assign_17^save_25/Assign_18^save_25/Assign_19^save_25/Assign_2^save_25/Assign_20^save_25/Assign_21^save_25/Assign_22^save_25/Assign_23^save_25/Assign_24^save_25/Assign_25^save_25/Assign_26^save_25/Assign_27^save_25/Assign_28^save_25/Assign_29^save_25/Assign_3^save_25/Assign_30^save_25/Assign_31^save_25/Assign_32^save_25/Assign_33^save_25/Assign_34^save_25/Assign_35^save_25/Assign_36^save_25/Assign_37^save_25/Assign_38^save_25/Assign_39^save_25/Assign_4^save_25/Assign_5^save_25/Assign_6^save_25/Assign_7^save_25/Assign_8^save_25/Assign_9
3
save_25/restore_allNoOp^save_25/restore_shard
\
save_26/filename/inputConst*
_output_shapes
: *
valueB Bmodel*
dtype0
t
save_26/filenamePlaceholderWithDefaultsave_26/filename/input*
_output_shapes
: *
dtype0*
shape: 
k
save_26/ConstPlaceholderWithDefaultsave_26/filename*
_output_shapes
: *
shape: *
dtype0
�
save_26/StringJoin/inputs_1Const*
dtype0*
_output_shapes
: *<
value3B1 B+_temp_2417404a8e9943b385c404039069bc1a/part
~
save_26/StringJoin
StringJoinsave_26/Constsave_26/StringJoin/inputs_1*
	separator *
_output_shapes
: *
N
T
save_26/num_shardsConst*
value	B :*
_output_shapes
: *
dtype0
_
save_26/ShardedFilename/shardConst*
value	B : *
_output_shapes
: *
dtype0
�
save_26/ShardedFilenameShardedFilenamesave_26/StringJoinsave_26/ShardedFilename/shardsave_26/num_shards*
_output_shapes
: 
�
save_26/SaveV2/tensor_namesConst*
dtype0*�
value�B�(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
_output_shapes
:(
�
save_26/SaveV2/shape_and_slicesConst*
_output_shapes
:(*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
�
save_26/SaveV2SaveV2save_26/ShardedFilenamesave_26/SaveV2/tensor_namessave_26/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1v/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1v/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1v/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1v/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1v/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1v/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1*6
dtypes,
*2(
�
save_26/control_dependencyIdentitysave_26/ShardedFilename^save_26/SaveV2*
_output_shapes
: **
_class 
loc:@save_26/ShardedFilename*
T0
�
.save_26/MergeV2Checkpoints/checkpoint_prefixesPacksave_26/ShardedFilename^save_26/control_dependency*
T0*
N*
_output_shapes
:*

axis 
�
save_26/MergeV2CheckpointsMergeV2Checkpoints.save_26/MergeV2Checkpoints/checkpoint_prefixessave_26/Const*
delete_old_dirs(
�
save_26/IdentityIdentitysave_26/Const^save_26/MergeV2Checkpoints^save_26/control_dependency*
T0*
_output_shapes
: 
�
save_26/RestoreV2/tensor_namesConst*
dtype0*�
value�B�(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
_output_shapes
:(
�
"save_26/RestoreV2/shape_and_slicesConst*
_output_shapes
:(*
dtype0*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
�
save_26/RestoreV2	RestoreV2save_26/Constsave_26/RestoreV2/tensor_names"save_26/RestoreV2/shape_and_slices*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::*6
dtypes,
*2(
�
save_26/AssignAssignbeta1_powersave_26/RestoreV2*
use_locking(*
validate_shape(*
T0* 
_class
loc:@pi/dense/bias*
_output_shapes
: 
�
save_26/Assign_1Assignbeta1_power_1save_26/RestoreV2:1*
_output_shapes
: *
use_locking(*
validate_shape(*
_class
loc:@v/dense/bias*
T0
�
save_26/Assign_2Assignbeta2_powersave_26/RestoreV2:2* 
_class
loc:@pi/dense/bias*
use_locking(*
validate_shape(*
_output_shapes
: *
T0
�
save_26/Assign_3Assignbeta2_power_1save_26/RestoreV2:3*
_class
loc:@v/dense/bias*
_output_shapes
: *
use_locking(*
validate_shape(*
T0
�
save_26/Assign_4Assignpi/dense/biassave_26/RestoreV2:4* 
_class
loc:@pi/dense/bias*
_output_shapes
:@*
validate_shape(*
T0*
use_locking(
�
save_26/Assign_5Assignpi/dense/bias/Adamsave_26/RestoreV2:5*
_output_shapes
:@*
validate_shape(* 
_class
loc:@pi/dense/bias*
use_locking(*
T0
�
save_26/Assign_6Assignpi/dense/bias/Adam_1save_26/RestoreV2:6*
_output_shapes
:@* 
_class
loc:@pi/dense/bias*
T0*
use_locking(*
validate_shape(
�
save_26/Assign_7Assignpi/dense/kernelsave_26/RestoreV2:7*
validate_shape(*
T0*
_output_shapes

:@*"
_class
loc:@pi/dense/kernel*
use_locking(
�
save_26/Assign_8Assignpi/dense/kernel/Adamsave_26/RestoreV2:8*
T0*
_output_shapes

:@*"
_class
loc:@pi/dense/kernel*
use_locking(*
validate_shape(
�
save_26/Assign_9Assignpi/dense/kernel/Adam_1save_26/RestoreV2:9*
_output_shapes

:@*
use_locking(*
validate_shape(*"
_class
loc:@pi/dense/kernel*
T0
�
save_26/Assign_10Assignpi/dense_1/biassave_26/RestoreV2:10*
_output_shapes
:@*
use_locking(*
validate_shape(*"
_class
loc:@pi/dense_1/bias*
T0
�
save_26/Assign_11Assignpi/dense_1/bias/Adamsave_26/RestoreV2:11*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
T0*
_output_shapes
:@*
use_locking(
�
save_26/Assign_12Assignpi/dense_1/bias/Adam_1save_26/RestoreV2:12*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
_output_shapes
:@*
T0*
use_locking(
�
save_26/Assign_13Assignpi/dense_1/kernelsave_26/RestoreV2:13*
T0*
use_locking(*$
_class
loc:@pi/dense_1/kernel*
_output_shapes

:@@*
validate_shape(
�
save_26/Assign_14Assignpi/dense_1/kernel/Adamsave_26/RestoreV2:14*
_output_shapes

:@@*$
_class
loc:@pi/dense_1/kernel*
T0*
use_locking(*
validate_shape(
�
save_26/Assign_15Assignpi/dense_1/kernel/Adam_1save_26/RestoreV2:15*
validate_shape(*
T0*
use_locking(*$
_class
loc:@pi/dense_1/kernel*
_output_shapes

:@@
�
save_26/Assign_16Assignpi/dense_2/biassave_26/RestoreV2:16*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
use_locking(*
T0*
_output_shapes
:
�
save_26/Assign_17Assignpi/dense_2/bias/Adamsave_26/RestoreV2:17*
use_locking(*
T0*
_output_shapes
:*"
_class
loc:@pi/dense_2/bias*
validate_shape(
�
save_26/Assign_18Assignpi/dense_2/bias/Adam_1save_26/RestoreV2:18*
validate_shape(*"
_class
loc:@pi/dense_2/bias*
T0*
_output_shapes
:*
use_locking(
�
save_26/Assign_19Assignpi/dense_2/kernelsave_26/RestoreV2:19*$
_class
loc:@pi/dense_2/kernel*
_output_shapes

:@*
validate_shape(*
use_locking(*
T0
�
save_26/Assign_20Assignpi/dense_2/kernel/Adamsave_26/RestoreV2:20*
use_locking(*
T0*$
_class
loc:@pi/dense_2/kernel*
_output_shapes

:@*
validate_shape(
�
save_26/Assign_21Assignpi/dense_2/kernel/Adam_1save_26/RestoreV2:21*
validate_shape(*
use_locking(*
_output_shapes

:@*$
_class
loc:@pi/dense_2/kernel*
T0
�
save_26/Assign_22Assignv/dense/biassave_26/RestoreV2:22*
use_locking(*
validate_shape(*
_class
loc:@v/dense/bias*
_output_shapes
:@*
T0
�
save_26/Assign_23Assignv/dense/bias/Adamsave_26/RestoreV2:23*
_output_shapes
:@*
validate_shape(*
_class
loc:@v/dense/bias*
T0*
use_locking(
�
save_26/Assign_24Assignv/dense/bias/Adam_1save_26/RestoreV2:24*
_output_shapes
:@*
validate_shape(*
use_locking(*
_class
loc:@v/dense/bias*
T0
�
save_26/Assign_25Assignv/dense/kernelsave_26/RestoreV2:25*!
_class
loc:@v/dense/kernel*
use_locking(*
_output_shapes

:@*
validate_shape(*
T0
�
save_26/Assign_26Assignv/dense/kernel/Adamsave_26/RestoreV2:26*
T0*
validate_shape(*
use_locking(*!
_class
loc:@v/dense/kernel*
_output_shapes

:@
�
save_26/Assign_27Assignv/dense/kernel/Adam_1save_26/RestoreV2:27*
validate_shape(*!
_class
loc:@v/dense/kernel*
_output_shapes

:@*
T0*
use_locking(
�
save_26/Assign_28Assignv/dense_1/biassave_26/RestoreV2:28*!
_class
loc:@v/dense_1/bias*
use_locking(*
validate_shape(*
T0*
_output_shapes
:@
�
save_26/Assign_29Assignv/dense_1/bias/Adamsave_26/RestoreV2:29*
T0*!
_class
loc:@v/dense_1/bias*
_output_shapes
:@*
validate_shape(*
use_locking(
�
save_26/Assign_30Assignv/dense_1/bias/Adam_1save_26/RestoreV2:30*
validate_shape(*!
_class
loc:@v/dense_1/bias*
_output_shapes
:@*
T0*
use_locking(
�
save_26/Assign_31Assignv/dense_1/kernelsave_26/RestoreV2:31*
_output_shapes

:@@*#
_class
loc:@v/dense_1/kernel*
validate_shape(*
use_locking(*
T0
�
save_26/Assign_32Assignv/dense_1/kernel/Adamsave_26/RestoreV2:32*#
_class
loc:@v/dense_1/kernel*
_output_shapes

:@@*
use_locking(*
T0*
validate_shape(
�
save_26/Assign_33Assignv/dense_1/kernel/Adam_1save_26/RestoreV2:33*
T0*
_output_shapes

:@@*
validate_shape(*
use_locking(*#
_class
loc:@v/dense_1/kernel
�
save_26/Assign_34Assignv/dense_2/biassave_26/RestoreV2:34*
_output_shapes
:*
use_locking(*
validate_shape(*!
_class
loc:@v/dense_2/bias*
T0
�
save_26/Assign_35Assignv/dense_2/bias/Adamsave_26/RestoreV2:35*!
_class
loc:@v/dense_2/bias*
_output_shapes
:*
validate_shape(*
use_locking(*
T0
�
save_26/Assign_36Assignv/dense_2/bias/Adam_1save_26/RestoreV2:36*
T0*
_output_shapes
:*
validate_shape(*!
_class
loc:@v/dense_2/bias*
use_locking(
�
save_26/Assign_37Assignv/dense_2/kernelsave_26/RestoreV2:37*
_output_shapes

:@*
use_locking(*
T0*#
_class
loc:@v/dense_2/kernel*
validate_shape(
�
save_26/Assign_38Assignv/dense_2/kernel/Adamsave_26/RestoreV2:38*#
_class
loc:@v/dense_2/kernel*
use_locking(*
T0*
_output_shapes

:@*
validate_shape(
�
save_26/Assign_39Assignv/dense_2/kernel/Adam_1save_26/RestoreV2:39*
_output_shapes

:@*#
_class
loc:@v/dense_2/kernel*
T0*
validate_shape(*
use_locking(
�
save_26/restore_shardNoOp^save_26/Assign^save_26/Assign_1^save_26/Assign_10^save_26/Assign_11^save_26/Assign_12^save_26/Assign_13^save_26/Assign_14^save_26/Assign_15^save_26/Assign_16^save_26/Assign_17^save_26/Assign_18^save_26/Assign_19^save_26/Assign_2^save_26/Assign_20^save_26/Assign_21^save_26/Assign_22^save_26/Assign_23^save_26/Assign_24^save_26/Assign_25^save_26/Assign_26^save_26/Assign_27^save_26/Assign_28^save_26/Assign_29^save_26/Assign_3^save_26/Assign_30^save_26/Assign_31^save_26/Assign_32^save_26/Assign_33^save_26/Assign_34^save_26/Assign_35^save_26/Assign_36^save_26/Assign_37^save_26/Assign_38^save_26/Assign_39^save_26/Assign_4^save_26/Assign_5^save_26/Assign_6^save_26/Assign_7^save_26/Assign_8^save_26/Assign_9
3
save_26/restore_allNoOp^save_26/restore_shard
\
save_27/filename/inputConst*
dtype0*
_output_shapes
: *
valueB Bmodel
t
save_27/filenamePlaceholderWithDefaultsave_27/filename/input*
shape: *
_output_shapes
: *
dtype0
k
save_27/ConstPlaceholderWithDefaultsave_27/filename*
shape: *
dtype0*
_output_shapes
: 
�
save_27/StringJoin/inputs_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+_temp_6506874e494a43d4b493a206d6dfeee4/part
~
save_27/StringJoin
StringJoinsave_27/Constsave_27/StringJoin/inputs_1*
_output_shapes
: *
	separator *
N
T
save_27/num_shardsConst*
_output_shapes
: *
value	B :*
dtype0
_
save_27/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 
�
save_27/ShardedFilenameShardedFilenamesave_27/StringJoinsave_27/ShardedFilename/shardsave_27/num_shards*
_output_shapes
: 
�
save_27/SaveV2/tensor_namesConst*
_output_shapes
:(*
dtype0*�
value�B�(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1
�
save_27/SaveV2/shape_and_slicesConst*
_output_shapes
:(*
dtype0*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
�
save_27/SaveV2SaveV2save_27/ShardedFilenamesave_27/SaveV2/tensor_namessave_27/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1v/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1v/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1v/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1v/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1v/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1v/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1*6
dtypes,
*2(
�
save_27/control_dependencyIdentitysave_27/ShardedFilename^save_27/SaveV2*
T0*
_output_shapes
: **
_class 
loc:@save_27/ShardedFilename
�
.save_27/MergeV2Checkpoints/checkpoint_prefixesPacksave_27/ShardedFilename^save_27/control_dependency*
_output_shapes
:*
N*
T0*

axis 
�
save_27/MergeV2CheckpointsMergeV2Checkpoints.save_27/MergeV2Checkpoints/checkpoint_prefixessave_27/Const*
delete_old_dirs(
�
save_27/IdentityIdentitysave_27/Const^save_27/MergeV2Checkpoints^save_27/control_dependency*
T0*
_output_shapes
: 
�
save_27/RestoreV2/tensor_namesConst*�
value�B�(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
_output_shapes
:(*
dtype0
�
"save_27/RestoreV2/shape_and_slicesConst*
dtype0*
_output_shapes
:(*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
�
save_27/RestoreV2	RestoreV2save_27/Constsave_27/RestoreV2/tensor_names"save_27/RestoreV2/shape_and_slices*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::*6
dtypes,
*2(
�
save_27/AssignAssignbeta1_powersave_27/RestoreV2*
T0*
use_locking(* 
_class
loc:@pi/dense/bias*
_output_shapes
: *
validate_shape(
�
save_27/Assign_1Assignbeta1_power_1save_27/RestoreV2:1*
validate_shape(*
_output_shapes
: *
_class
loc:@v/dense/bias*
use_locking(*
T0
�
save_27/Assign_2Assignbeta2_powersave_27/RestoreV2:2* 
_class
loc:@pi/dense/bias*
_output_shapes
: *
validate_shape(*
T0*
use_locking(
�
save_27/Assign_3Assignbeta2_power_1save_27/RestoreV2:3*
_output_shapes
: *
validate_shape(*
use_locking(*
T0*
_class
loc:@v/dense/bias
�
save_27/Assign_4Assignpi/dense/biassave_27/RestoreV2:4*
_output_shapes
:@*
use_locking(*
validate_shape(* 
_class
loc:@pi/dense/bias*
T0
�
save_27/Assign_5Assignpi/dense/bias/Adamsave_27/RestoreV2:5*
_output_shapes
:@* 
_class
loc:@pi/dense/bias*
use_locking(*
T0*
validate_shape(
�
save_27/Assign_6Assignpi/dense/bias/Adam_1save_27/RestoreV2:6*
T0*
use_locking(*
_output_shapes
:@*
validate_shape(* 
_class
loc:@pi/dense/bias
�
save_27/Assign_7Assignpi/dense/kernelsave_27/RestoreV2:7*
_output_shapes

:@*
T0*
use_locking(*
validate_shape(*"
_class
loc:@pi/dense/kernel
�
save_27/Assign_8Assignpi/dense/kernel/Adamsave_27/RestoreV2:8*
T0*
_output_shapes

:@*"
_class
loc:@pi/dense/kernel*
validate_shape(*
use_locking(
�
save_27/Assign_9Assignpi/dense/kernel/Adam_1save_27/RestoreV2:9*"
_class
loc:@pi/dense/kernel*
T0*
_output_shapes

:@*
validate_shape(*
use_locking(
�
save_27/Assign_10Assignpi/dense_1/biassave_27/RestoreV2:10*"
_class
loc:@pi/dense_1/bias*
_output_shapes
:@*
validate_shape(*
use_locking(*
T0
�
save_27/Assign_11Assignpi/dense_1/bias/Adamsave_27/RestoreV2:11*"
_class
loc:@pi/dense_1/bias*
T0*
use_locking(*
validate_shape(*
_output_shapes
:@
�
save_27/Assign_12Assignpi/dense_1/bias/Adam_1save_27/RestoreV2:12*"
_class
loc:@pi/dense_1/bias*
_output_shapes
:@*
use_locking(*
validate_shape(*
T0
�
save_27/Assign_13Assignpi/dense_1/kernelsave_27/RestoreV2:13*
validate_shape(*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel*
_output_shapes

:@@
�
save_27/Assign_14Assignpi/dense_1/kernel/Adamsave_27/RestoreV2:14*
T0*
_output_shapes

:@@*
use_locking(*$
_class
loc:@pi/dense_1/kernel*
validate_shape(
�
save_27/Assign_15Assignpi/dense_1/kernel/Adam_1save_27/RestoreV2:15*$
_class
loc:@pi/dense_1/kernel*
_output_shapes

:@@*
T0*
use_locking(*
validate_shape(
�
save_27/Assign_16Assignpi/dense_2/biassave_27/RestoreV2:16*"
_class
loc:@pi/dense_2/bias*
T0*
use_locking(*
validate_shape(*
_output_shapes
:
�
save_27/Assign_17Assignpi/dense_2/bias/Adamsave_27/RestoreV2:17*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:*
validate_shape(*
T0*
use_locking(
�
save_27/Assign_18Assignpi/dense_2/bias/Adam_1save_27/RestoreV2:18*
_output_shapes
:*
validate_shape(*
T0*"
_class
loc:@pi/dense_2/bias*
use_locking(
�
save_27/Assign_19Assignpi/dense_2/kernelsave_27/RestoreV2:19*
use_locking(*
_output_shapes

:@*$
_class
loc:@pi/dense_2/kernel*
T0*
validate_shape(
�
save_27/Assign_20Assignpi/dense_2/kernel/Adamsave_27/RestoreV2:20*
use_locking(*
_output_shapes

:@*
validate_shape(*$
_class
loc:@pi/dense_2/kernel*
T0
�
save_27/Assign_21Assignpi/dense_2/kernel/Adam_1save_27/RestoreV2:21*$
_class
loc:@pi/dense_2/kernel*
T0*
_output_shapes

:@*
validate_shape(*
use_locking(
�
save_27/Assign_22Assignv/dense/biassave_27/RestoreV2:22*
T0*
_class
loc:@v/dense/bias*
_output_shapes
:@*
validate_shape(*
use_locking(
�
save_27/Assign_23Assignv/dense/bias/Adamsave_27/RestoreV2:23*
_output_shapes
:@*
T0*
use_locking(*
validate_shape(*
_class
loc:@v/dense/bias
�
save_27/Assign_24Assignv/dense/bias/Adam_1save_27/RestoreV2:24*
_output_shapes
:@*
validate_shape(*
use_locking(*
T0*
_class
loc:@v/dense/bias
�
save_27/Assign_25Assignv/dense/kernelsave_27/RestoreV2:25*
use_locking(*!
_class
loc:@v/dense/kernel*
validate_shape(*
_output_shapes

:@*
T0
�
save_27/Assign_26Assignv/dense/kernel/Adamsave_27/RestoreV2:26*
_output_shapes

:@*
use_locking(*!
_class
loc:@v/dense/kernel*
validate_shape(*
T0
�
save_27/Assign_27Assignv/dense/kernel/Adam_1save_27/RestoreV2:27*
validate_shape(*!
_class
loc:@v/dense/kernel*
use_locking(*
_output_shapes

:@*
T0
�
save_27/Assign_28Assignv/dense_1/biassave_27/RestoreV2:28*!
_class
loc:@v/dense_1/bias*
_output_shapes
:@*
use_locking(*
T0*
validate_shape(
�
save_27/Assign_29Assignv/dense_1/bias/Adamsave_27/RestoreV2:29*
_output_shapes
:@*!
_class
loc:@v/dense_1/bias*
T0*
use_locking(*
validate_shape(
�
save_27/Assign_30Assignv/dense_1/bias/Adam_1save_27/RestoreV2:30*
validate_shape(*
use_locking(*
T0*!
_class
loc:@v/dense_1/bias*
_output_shapes
:@
�
save_27/Assign_31Assignv/dense_1/kernelsave_27/RestoreV2:31*#
_class
loc:@v/dense_1/kernel*
_output_shapes

:@@*
T0*
use_locking(*
validate_shape(
�
save_27/Assign_32Assignv/dense_1/kernel/Adamsave_27/RestoreV2:32*
validate_shape(*
T0*
_output_shapes

:@@*
use_locking(*#
_class
loc:@v/dense_1/kernel
�
save_27/Assign_33Assignv/dense_1/kernel/Adam_1save_27/RestoreV2:33*
T0*
_output_shapes

:@@*#
_class
loc:@v/dense_1/kernel*
validate_shape(*
use_locking(
�
save_27/Assign_34Assignv/dense_2/biassave_27/RestoreV2:34*
T0*!
_class
loc:@v/dense_2/bias*
_output_shapes
:*
use_locking(*
validate_shape(
�
save_27/Assign_35Assignv/dense_2/bias/Adamsave_27/RestoreV2:35*
_output_shapes
:*
T0*
validate_shape(*
use_locking(*!
_class
loc:@v/dense_2/bias
�
save_27/Assign_36Assignv/dense_2/bias/Adam_1save_27/RestoreV2:36*
_output_shapes
:*
validate_shape(*
use_locking(*
T0*!
_class
loc:@v/dense_2/bias
�
save_27/Assign_37Assignv/dense_2/kernelsave_27/RestoreV2:37*
validate_shape(*
use_locking(*
_output_shapes

:@*
T0*#
_class
loc:@v/dense_2/kernel
�
save_27/Assign_38Assignv/dense_2/kernel/Adamsave_27/RestoreV2:38*
use_locking(*
validate_shape(*
T0*
_output_shapes

:@*#
_class
loc:@v/dense_2/kernel
�
save_27/Assign_39Assignv/dense_2/kernel/Adam_1save_27/RestoreV2:39*
_output_shapes

:@*
T0*
validate_shape(*
use_locking(*#
_class
loc:@v/dense_2/kernel
�
save_27/restore_shardNoOp^save_27/Assign^save_27/Assign_1^save_27/Assign_10^save_27/Assign_11^save_27/Assign_12^save_27/Assign_13^save_27/Assign_14^save_27/Assign_15^save_27/Assign_16^save_27/Assign_17^save_27/Assign_18^save_27/Assign_19^save_27/Assign_2^save_27/Assign_20^save_27/Assign_21^save_27/Assign_22^save_27/Assign_23^save_27/Assign_24^save_27/Assign_25^save_27/Assign_26^save_27/Assign_27^save_27/Assign_28^save_27/Assign_29^save_27/Assign_3^save_27/Assign_30^save_27/Assign_31^save_27/Assign_32^save_27/Assign_33^save_27/Assign_34^save_27/Assign_35^save_27/Assign_36^save_27/Assign_37^save_27/Assign_38^save_27/Assign_39^save_27/Assign_4^save_27/Assign_5^save_27/Assign_6^save_27/Assign_7^save_27/Assign_8^save_27/Assign_9
3
save_27/restore_allNoOp^save_27/restore_shard
\
save_28/filename/inputConst*
_output_shapes
: *
dtype0*
valueB Bmodel
t
save_28/filenamePlaceholderWithDefaultsave_28/filename/input*
_output_shapes
: *
dtype0*
shape: 
k
save_28/ConstPlaceholderWithDefaultsave_28/filename*
_output_shapes
: *
dtype0*
shape: 
�
save_28/StringJoin/inputs_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+_temp_aa76d04528c2435a9e875ac64e2d066f/part
~
save_28/StringJoin
StringJoinsave_28/Constsave_28/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
T
save_28/num_shardsConst*
_output_shapes
: *
dtype0*
value	B :
_
save_28/ShardedFilename/shardConst*
_output_shapes
: *
value	B : *
dtype0
�
save_28/ShardedFilenameShardedFilenamesave_28/StringJoinsave_28/ShardedFilename/shardsave_28/num_shards*
_output_shapes
: 
�
save_28/SaveV2/tensor_namesConst*�
value�B�(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:(
�
save_28/SaveV2/shape_and_slicesConst*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:(
�
save_28/SaveV2SaveV2save_28/ShardedFilenamesave_28/SaveV2/tensor_namessave_28/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1v/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1v/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1v/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1v/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1v/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1v/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1*6
dtypes,
*2(
�
save_28/control_dependencyIdentitysave_28/ShardedFilename^save_28/SaveV2**
_class 
loc:@save_28/ShardedFilename*
T0*
_output_shapes
: 
�
.save_28/MergeV2Checkpoints/checkpoint_prefixesPacksave_28/ShardedFilename^save_28/control_dependency*
N*
_output_shapes
:*
T0*

axis 
�
save_28/MergeV2CheckpointsMergeV2Checkpoints.save_28/MergeV2Checkpoints/checkpoint_prefixessave_28/Const*
delete_old_dirs(
�
save_28/IdentityIdentitysave_28/Const^save_28/MergeV2Checkpoints^save_28/control_dependency*
T0*
_output_shapes
: 
�
save_28/RestoreV2/tensor_namesConst*�
value�B�(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
_output_shapes
:(*
dtype0
�
"save_28/RestoreV2/shape_and_slicesConst*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:(
�
save_28/RestoreV2	RestoreV2save_28/Constsave_28/RestoreV2/tensor_names"save_28/RestoreV2/shape_and_slices*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::*6
dtypes,
*2(
�
save_28/AssignAssignbeta1_powersave_28/RestoreV2*
use_locking(*
validate_shape(* 
_class
loc:@pi/dense/bias*
T0*
_output_shapes
: 
�
save_28/Assign_1Assignbeta1_power_1save_28/RestoreV2:1*
validate_shape(*
_class
loc:@v/dense/bias*
use_locking(*
T0*
_output_shapes
: 
�
save_28/Assign_2Assignbeta2_powersave_28/RestoreV2:2*
use_locking(*
_output_shapes
: *
validate_shape(*
T0* 
_class
loc:@pi/dense/bias
�
save_28/Assign_3Assignbeta2_power_1save_28/RestoreV2:3*
T0*
use_locking(*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
: 
�
save_28/Assign_4Assignpi/dense/biassave_28/RestoreV2:4* 
_class
loc:@pi/dense/bias*
_output_shapes
:@*
use_locking(*
T0*
validate_shape(
�
save_28/Assign_5Assignpi/dense/bias/Adamsave_28/RestoreV2:5* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
:@*
T0*
use_locking(
�
save_28/Assign_6Assignpi/dense/bias/Adam_1save_28/RestoreV2:6* 
_class
loc:@pi/dense/bias*
T0*
validate_shape(*
_output_shapes
:@*
use_locking(
�
save_28/Assign_7Assignpi/dense/kernelsave_28/RestoreV2:7*
use_locking(*
_output_shapes

:@*
validate_shape(*
T0*"
_class
loc:@pi/dense/kernel
�
save_28/Assign_8Assignpi/dense/kernel/Adamsave_28/RestoreV2:8*
validate_shape(*
use_locking(*
T0*
_output_shapes

:@*"
_class
loc:@pi/dense/kernel
�
save_28/Assign_9Assignpi/dense/kernel/Adam_1save_28/RestoreV2:9*
use_locking(*
validate_shape(*
T0*"
_class
loc:@pi/dense/kernel*
_output_shapes

:@
�
save_28/Assign_10Assignpi/dense_1/biassave_28/RestoreV2:10*
T0*
validate_shape(*"
_class
loc:@pi/dense_1/bias*
use_locking(*
_output_shapes
:@
�
save_28/Assign_11Assignpi/dense_1/bias/Adamsave_28/RestoreV2:11*
T0*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
_output_shapes
:@*
use_locking(
�
save_28/Assign_12Assignpi/dense_1/bias/Adam_1save_28/RestoreV2:12*
use_locking(*
T0*"
_class
loc:@pi/dense_1/bias*
_output_shapes
:@*
validate_shape(
�
save_28/Assign_13Assignpi/dense_1/kernelsave_28/RestoreV2:13*
validate_shape(*
_output_shapes

:@@*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel
�
save_28/Assign_14Assignpi/dense_1/kernel/Adamsave_28/RestoreV2:14*
use_locking(*
_output_shapes

:@@*$
_class
loc:@pi/dense_1/kernel*
validate_shape(*
T0
�
save_28/Assign_15Assignpi/dense_1/kernel/Adam_1save_28/RestoreV2:15*
_output_shapes

:@@*$
_class
loc:@pi/dense_1/kernel*
use_locking(*
T0*
validate_shape(
�
save_28/Assign_16Assignpi/dense_2/biassave_28/RestoreV2:16*
use_locking(*
validate_shape(*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:*
T0
�
save_28/Assign_17Assignpi/dense_2/bias/Adamsave_28/RestoreV2:17*
_output_shapes
:*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
use_locking(*
T0
�
save_28/Assign_18Assignpi/dense_2/bias/Adam_1save_28/RestoreV2:18*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:*
validate_shape(*
T0*
use_locking(
�
save_28/Assign_19Assignpi/dense_2/kernelsave_28/RestoreV2:19*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
T0*
_output_shapes

:@*
use_locking(
�
save_28/Assign_20Assignpi/dense_2/kernel/Adamsave_28/RestoreV2:20*
validate_shape(*
T0*$
_class
loc:@pi/dense_2/kernel*
use_locking(*
_output_shapes

:@
�
save_28/Assign_21Assignpi/dense_2/kernel/Adam_1save_28/RestoreV2:21*
validate_shape(*
use_locking(*
T0*
_output_shapes

:@*$
_class
loc:@pi/dense_2/kernel
�
save_28/Assign_22Assignv/dense/biassave_28/RestoreV2:22*
T0*
_output_shapes
:@*
validate_shape(*
_class
loc:@v/dense/bias*
use_locking(
�
save_28/Assign_23Assignv/dense/bias/Adamsave_28/RestoreV2:23*
_output_shapes
:@*
validate_shape(*
T0*
use_locking(*
_class
loc:@v/dense/bias
�
save_28/Assign_24Assignv/dense/bias/Adam_1save_28/RestoreV2:24*
use_locking(*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
:@*
T0
�
save_28/Assign_25Assignv/dense/kernelsave_28/RestoreV2:25*!
_class
loc:@v/dense/kernel*
T0*
_output_shapes

:@*
validate_shape(*
use_locking(
�
save_28/Assign_26Assignv/dense/kernel/Adamsave_28/RestoreV2:26*
use_locking(*
_output_shapes

:@*!
_class
loc:@v/dense/kernel*
validate_shape(*
T0
�
save_28/Assign_27Assignv/dense/kernel/Adam_1save_28/RestoreV2:27*
T0*
validate_shape(*
_output_shapes

:@*
use_locking(*!
_class
loc:@v/dense/kernel
�
save_28/Assign_28Assignv/dense_1/biassave_28/RestoreV2:28*!
_class
loc:@v/dense_1/bias*
_output_shapes
:@*
validate_shape(*
T0*
use_locking(
�
save_28/Assign_29Assignv/dense_1/bias/Adamsave_28/RestoreV2:29*
_output_shapes
:@*!
_class
loc:@v/dense_1/bias*
use_locking(*
T0*
validate_shape(
�
save_28/Assign_30Assignv/dense_1/bias/Adam_1save_28/RestoreV2:30*
validate_shape(*!
_class
loc:@v/dense_1/bias*
_output_shapes
:@*
T0*
use_locking(
�
save_28/Assign_31Assignv/dense_1/kernelsave_28/RestoreV2:31*
T0*#
_class
loc:@v/dense_1/kernel*
validate_shape(*
use_locking(*
_output_shapes

:@@
�
save_28/Assign_32Assignv/dense_1/kernel/Adamsave_28/RestoreV2:32*#
_class
loc:@v/dense_1/kernel*
_output_shapes

:@@*
T0*
use_locking(*
validate_shape(
�
save_28/Assign_33Assignv/dense_1/kernel/Adam_1save_28/RestoreV2:33*
use_locking(*
validate_shape(*#
_class
loc:@v/dense_1/kernel*
_output_shapes

:@@*
T0
�
save_28/Assign_34Assignv/dense_2/biassave_28/RestoreV2:34*
use_locking(*
_output_shapes
:*
validate_shape(*
T0*!
_class
loc:@v/dense_2/bias
�
save_28/Assign_35Assignv/dense_2/bias/Adamsave_28/RestoreV2:35*
T0*!
_class
loc:@v/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(
�
save_28/Assign_36Assignv/dense_2/bias/Adam_1save_28/RestoreV2:36*
use_locking(*!
_class
loc:@v/dense_2/bias*
validate_shape(*
_output_shapes
:*
T0
�
save_28/Assign_37Assignv/dense_2/kernelsave_28/RestoreV2:37*
validate_shape(*
use_locking(*
T0*
_output_shapes

:@*#
_class
loc:@v/dense_2/kernel
�
save_28/Assign_38Assignv/dense_2/kernel/Adamsave_28/RestoreV2:38*
T0*
validate_shape(*
_output_shapes

:@*#
_class
loc:@v/dense_2/kernel*
use_locking(
�
save_28/Assign_39Assignv/dense_2/kernel/Adam_1save_28/RestoreV2:39*#
_class
loc:@v/dense_2/kernel*
use_locking(*
validate_shape(*
_output_shapes

:@*
T0
�
save_28/restore_shardNoOp^save_28/Assign^save_28/Assign_1^save_28/Assign_10^save_28/Assign_11^save_28/Assign_12^save_28/Assign_13^save_28/Assign_14^save_28/Assign_15^save_28/Assign_16^save_28/Assign_17^save_28/Assign_18^save_28/Assign_19^save_28/Assign_2^save_28/Assign_20^save_28/Assign_21^save_28/Assign_22^save_28/Assign_23^save_28/Assign_24^save_28/Assign_25^save_28/Assign_26^save_28/Assign_27^save_28/Assign_28^save_28/Assign_29^save_28/Assign_3^save_28/Assign_30^save_28/Assign_31^save_28/Assign_32^save_28/Assign_33^save_28/Assign_34^save_28/Assign_35^save_28/Assign_36^save_28/Assign_37^save_28/Assign_38^save_28/Assign_39^save_28/Assign_4^save_28/Assign_5^save_28/Assign_6^save_28/Assign_7^save_28/Assign_8^save_28/Assign_9
3
save_28/restore_allNoOp^save_28/restore_shard
\
save_29/filename/inputConst*
dtype0*
valueB Bmodel*
_output_shapes
: 
t
save_29/filenamePlaceholderWithDefaultsave_29/filename/input*
shape: *
_output_shapes
: *
dtype0
k
save_29/ConstPlaceholderWithDefaultsave_29/filename*
dtype0*
_output_shapes
: *
shape: 
�
save_29/StringJoin/inputs_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+_temp_a29b0bcf2027462dae0e01613f9ebb62/part
~
save_29/StringJoin
StringJoinsave_29/Constsave_29/StringJoin/inputs_1*
	separator *
N*
_output_shapes
: 
T
save_29/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
_
save_29/ShardedFilename/shardConst*
value	B : *
_output_shapes
: *
dtype0
�
save_29/ShardedFilenameShardedFilenamesave_29/StringJoinsave_29/ShardedFilename/shardsave_29/num_shards*
_output_shapes
: 
�
save_29/SaveV2/tensor_namesConst*
dtype0*
_output_shapes
:(*�
value�B�(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1
�
save_29/SaveV2/shape_and_slicesConst*
_output_shapes
:(*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
�
save_29/SaveV2SaveV2save_29/ShardedFilenamesave_29/SaveV2/tensor_namessave_29/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1v/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1v/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1v/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1v/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1v/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1v/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1*6
dtypes,
*2(
�
save_29/control_dependencyIdentitysave_29/ShardedFilename^save_29/SaveV2*
T0*
_output_shapes
: **
_class 
loc:@save_29/ShardedFilename
�
.save_29/MergeV2Checkpoints/checkpoint_prefixesPacksave_29/ShardedFilename^save_29/control_dependency*

axis *
T0*
_output_shapes
:*
N
�
save_29/MergeV2CheckpointsMergeV2Checkpoints.save_29/MergeV2Checkpoints/checkpoint_prefixessave_29/Const*
delete_old_dirs(
�
save_29/IdentityIdentitysave_29/Const^save_29/MergeV2Checkpoints^save_29/control_dependency*
T0*
_output_shapes
: 
�
save_29/RestoreV2/tensor_namesConst*
dtype0*�
value�B�(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
_output_shapes
:(
�
"save_29/RestoreV2/shape_and_slicesConst*
dtype0*
_output_shapes
:(*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
�
save_29/RestoreV2	RestoreV2save_29/Constsave_29/RestoreV2/tensor_names"save_29/RestoreV2/shape_and_slices*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::*6
dtypes,
*2(
�
save_29/AssignAssignbeta1_powersave_29/RestoreV2* 
_class
loc:@pi/dense/bias*
_output_shapes
: *
validate_shape(*
T0*
use_locking(
�
save_29/Assign_1Assignbeta1_power_1save_29/RestoreV2:1*
_output_shapes
: *
validate_shape(*
use_locking(*
T0*
_class
loc:@v/dense/bias
�
save_29/Assign_2Assignbeta2_powersave_29/RestoreV2:2*
_output_shapes
: *
T0*
validate_shape(*
use_locking(* 
_class
loc:@pi/dense/bias
�
save_29/Assign_3Assignbeta2_power_1save_29/RestoreV2:3*
_output_shapes
: *
use_locking(*
validate_shape(*
T0*
_class
loc:@v/dense/bias
�
save_29/Assign_4Assignpi/dense/biassave_29/RestoreV2:4*
use_locking(*
_output_shapes
:@*
T0*
validate_shape(* 
_class
loc:@pi/dense/bias
�
save_29/Assign_5Assignpi/dense/bias/Adamsave_29/RestoreV2:5*
validate_shape(*
_output_shapes
:@*
T0*
use_locking(* 
_class
loc:@pi/dense/bias
�
save_29/Assign_6Assignpi/dense/bias/Adam_1save_29/RestoreV2:6*
T0* 
_class
loc:@pi/dense/bias*
use_locking(*
_output_shapes
:@*
validate_shape(
�
save_29/Assign_7Assignpi/dense/kernelsave_29/RestoreV2:7*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0
�
save_29/Assign_8Assignpi/dense/kernel/Adamsave_29/RestoreV2:8*
_output_shapes

:@*"
_class
loc:@pi/dense/kernel*
T0*
validate_shape(*
use_locking(
�
save_29/Assign_9Assignpi/dense/kernel/Adam_1save_29/RestoreV2:9*"
_class
loc:@pi/dense/kernel*
use_locking(*
validate_shape(*
_output_shapes

:@*
T0
�
save_29/Assign_10Assignpi/dense_1/biassave_29/RestoreV2:10*
_output_shapes
:@*
use_locking(*"
_class
loc:@pi/dense_1/bias*
T0*
validate_shape(
�
save_29/Assign_11Assignpi/dense_1/bias/Adamsave_29/RestoreV2:11*
T0*
_output_shapes
:@*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
use_locking(
�
save_29/Assign_12Assignpi/dense_1/bias/Adam_1save_29/RestoreV2:12*
_output_shapes
:@*"
_class
loc:@pi/dense_1/bias*
use_locking(*
T0*
validate_shape(
�
save_29/Assign_13Assignpi/dense_1/kernelsave_29/RestoreV2:13*$
_class
loc:@pi/dense_1/kernel*
_output_shapes

:@@*
validate_shape(*
use_locking(*
T0
�
save_29/Assign_14Assignpi/dense_1/kernel/Adamsave_29/RestoreV2:14*
T0*
use_locking(*$
_class
loc:@pi/dense_1/kernel*
_output_shapes

:@@*
validate_shape(
�
save_29/Assign_15Assignpi/dense_1/kernel/Adam_1save_29/RestoreV2:15*
T0*
validate_shape(*$
_class
loc:@pi/dense_1/kernel*
use_locking(*
_output_shapes

:@@
�
save_29/Assign_16Assignpi/dense_2/biassave_29/RestoreV2:16*
validate_shape(*
T0*
use_locking(*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:
�
save_29/Assign_17Assignpi/dense_2/bias/Adamsave_29/RestoreV2:17*
use_locking(*
validate_shape(*
T0*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:
�
save_29/Assign_18Assignpi/dense_2/bias/Adam_1save_29/RestoreV2:18*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:*
use_locking(*
T0*
validate_shape(
�
save_29/Assign_19Assignpi/dense_2/kernelsave_29/RestoreV2:19*
use_locking(*
validate_shape(*
T0*
_output_shapes

:@*$
_class
loc:@pi/dense_2/kernel
�
save_29/Assign_20Assignpi/dense_2/kernel/Adamsave_29/RestoreV2:20*
T0*
use_locking(*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@
�
save_29/Assign_21Assignpi/dense_2/kernel/Adam_1save_29/RestoreV2:21*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0*$
_class
loc:@pi/dense_2/kernel
�
save_29/Assign_22Assignv/dense/biassave_29/RestoreV2:22*
T0*
_class
loc:@v/dense/bias*
use_locking(*
_output_shapes
:@*
validate_shape(
�
save_29/Assign_23Assignv/dense/bias/Adamsave_29/RestoreV2:23*
_output_shapes
:@*
_class
loc:@v/dense/bias*
validate_shape(*
use_locking(*
T0
�
save_29/Assign_24Assignv/dense/bias/Adam_1save_29/RestoreV2:24*
T0*
_output_shapes
:@*
validate_shape(*
_class
loc:@v/dense/bias*
use_locking(
�
save_29/Assign_25Assignv/dense/kernelsave_29/RestoreV2:25*!
_class
loc:@v/dense/kernel*
use_locking(*
_output_shapes

:@*
T0*
validate_shape(
�
save_29/Assign_26Assignv/dense/kernel/Adamsave_29/RestoreV2:26*
use_locking(*
T0*
validate_shape(*!
_class
loc:@v/dense/kernel*
_output_shapes

:@
�
save_29/Assign_27Assignv/dense/kernel/Adam_1save_29/RestoreV2:27*
_output_shapes

:@*
use_locking(*
T0*
validate_shape(*!
_class
loc:@v/dense/kernel
�
save_29/Assign_28Assignv/dense_1/biassave_29/RestoreV2:28*!
_class
loc:@v/dense_1/bias*
use_locking(*
validate_shape(*
_output_shapes
:@*
T0
�
save_29/Assign_29Assignv/dense_1/bias/Adamsave_29/RestoreV2:29*
validate_shape(*
T0*
use_locking(*!
_class
loc:@v/dense_1/bias*
_output_shapes
:@
�
save_29/Assign_30Assignv/dense_1/bias/Adam_1save_29/RestoreV2:30*
_output_shapes
:@*
T0*
validate_shape(*!
_class
loc:@v/dense_1/bias*
use_locking(
�
save_29/Assign_31Assignv/dense_1/kernelsave_29/RestoreV2:31*
validate_shape(*
T0*
use_locking(*#
_class
loc:@v/dense_1/kernel*
_output_shapes

:@@
�
save_29/Assign_32Assignv/dense_1/kernel/Adamsave_29/RestoreV2:32*#
_class
loc:@v/dense_1/kernel*
_output_shapes

:@@*
T0*
use_locking(*
validate_shape(
�
save_29/Assign_33Assignv/dense_1/kernel/Adam_1save_29/RestoreV2:33*
T0*#
_class
loc:@v/dense_1/kernel*
_output_shapes

:@@*
use_locking(*
validate_shape(
�
save_29/Assign_34Assignv/dense_2/biassave_29/RestoreV2:34*
T0*
use_locking(*
_output_shapes
:*!
_class
loc:@v/dense_2/bias*
validate_shape(
�
save_29/Assign_35Assignv/dense_2/bias/Adamsave_29/RestoreV2:35*
use_locking(*
T0*!
_class
loc:@v/dense_2/bias*
validate_shape(*
_output_shapes
:
�
save_29/Assign_36Assignv/dense_2/bias/Adam_1save_29/RestoreV2:36*
T0*
_output_shapes
:*!
_class
loc:@v/dense_2/bias*
validate_shape(*
use_locking(
�
save_29/Assign_37Assignv/dense_2/kernelsave_29/RestoreV2:37*
T0*
_output_shapes

:@*
use_locking(*
validate_shape(*#
_class
loc:@v/dense_2/kernel
�
save_29/Assign_38Assignv/dense_2/kernel/Adamsave_29/RestoreV2:38*#
_class
loc:@v/dense_2/kernel*
_output_shapes

:@*
use_locking(*
T0*
validate_shape(
�
save_29/Assign_39Assignv/dense_2/kernel/Adam_1save_29/RestoreV2:39*
validate_shape(*
_output_shapes

:@*#
_class
loc:@v/dense_2/kernel*
use_locking(*
T0
�
save_29/restore_shardNoOp^save_29/Assign^save_29/Assign_1^save_29/Assign_10^save_29/Assign_11^save_29/Assign_12^save_29/Assign_13^save_29/Assign_14^save_29/Assign_15^save_29/Assign_16^save_29/Assign_17^save_29/Assign_18^save_29/Assign_19^save_29/Assign_2^save_29/Assign_20^save_29/Assign_21^save_29/Assign_22^save_29/Assign_23^save_29/Assign_24^save_29/Assign_25^save_29/Assign_26^save_29/Assign_27^save_29/Assign_28^save_29/Assign_29^save_29/Assign_3^save_29/Assign_30^save_29/Assign_31^save_29/Assign_32^save_29/Assign_33^save_29/Assign_34^save_29/Assign_35^save_29/Assign_36^save_29/Assign_37^save_29/Assign_38^save_29/Assign_39^save_29/Assign_4^save_29/Assign_5^save_29/Assign_6^save_29/Assign_7^save_29/Assign_8^save_29/Assign_9
3
save_29/restore_allNoOp^save_29/restore_shard
\
save_30/filename/inputConst*
dtype0*
valueB Bmodel*
_output_shapes
: 
t
save_30/filenamePlaceholderWithDefaultsave_30/filename/input*
shape: *
dtype0*
_output_shapes
: 
k
save_30/ConstPlaceholderWithDefaultsave_30/filename*
dtype0*
shape: *
_output_shapes
: 
�
save_30/StringJoin/inputs_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+_temp_f34521d5aa90437ab0c8a1ab08773b6d/part
~
save_30/StringJoin
StringJoinsave_30/Constsave_30/StringJoin/inputs_1*
_output_shapes
: *
	separator *
N
T
save_30/num_shardsConst*
dtype0*
_output_shapes
: *
value	B :
_
save_30/ShardedFilename/shardConst*
_output_shapes
: *
value	B : *
dtype0
�
save_30/ShardedFilenameShardedFilenamesave_30/StringJoinsave_30/ShardedFilename/shardsave_30/num_shards*
_output_shapes
: 
�
save_30/SaveV2/tensor_namesConst*�
value�B�(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
_output_shapes
:(*
dtype0
�
save_30/SaveV2/shape_and_slicesConst*
dtype0*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:(
�
save_30/SaveV2SaveV2save_30/ShardedFilenamesave_30/SaveV2/tensor_namessave_30/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1v/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1v/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1v/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1v/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1v/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1v/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1*6
dtypes,
*2(
�
save_30/control_dependencyIdentitysave_30/ShardedFilename^save_30/SaveV2**
_class 
loc:@save_30/ShardedFilename*
T0*
_output_shapes
: 
�
.save_30/MergeV2Checkpoints/checkpoint_prefixesPacksave_30/ShardedFilename^save_30/control_dependency*
T0*

axis *
N*
_output_shapes
:
�
save_30/MergeV2CheckpointsMergeV2Checkpoints.save_30/MergeV2Checkpoints/checkpoint_prefixessave_30/Const*
delete_old_dirs(
�
save_30/IdentityIdentitysave_30/Const^save_30/MergeV2Checkpoints^save_30/control_dependency*
T0*
_output_shapes
: 
�
save_30/RestoreV2/tensor_namesConst*�
value�B�(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:(
�
"save_30/RestoreV2/shape_and_slicesConst*
_output_shapes
:(*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
�
save_30/RestoreV2	RestoreV2save_30/Constsave_30/RestoreV2/tensor_names"save_30/RestoreV2/shape_and_slices*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::*6
dtypes,
*2(
�
save_30/AssignAssignbeta1_powersave_30/RestoreV2*
T0*
use_locking(* 
_class
loc:@pi/dense/bias*
_output_shapes
: *
validate_shape(
�
save_30/Assign_1Assignbeta1_power_1save_30/RestoreV2:1*
_output_shapes
: *
use_locking(*
_class
loc:@v/dense/bias*
validate_shape(*
T0
�
save_30/Assign_2Assignbeta2_powersave_30/RestoreV2:2*
validate_shape(* 
_class
loc:@pi/dense/bias*
T0*
_output_shapes
: *
use_locking(
�
save_30/Assign_3Assignbeta2_power_1save_30/RestoreV2:3*
validate_shape(*
_output_shapes
: *
_class
loc:@v/dense/bias*
use_locking(*
T0
�
save_30/Assign_4Assignpi/dense/biassave_30/RestoreV2:4*
_output_shapes
:@*
validate_shape(*
T0* 
_class
loc:@pi/dense/bias*
use_locking(
�
save_30/Assign_5Assignpi/dense/bias/Adamsave_30/RestoreV2:5*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
:@*
use_locking(
�
save_30/Assign_6Assignpi/dense/bias/Adam_1save_30/RestoreV2:6* 
_class
loc:@pi/dense/bias*
use_locking(*
validate_shape(*
_output_shapes
:@*
T0
�
save_30/Assign_7Assignpi/dense/kernelsave_30/RestoreV2:7*
validate_shape(*
use_locking(*
_output_shapes

:@*
T0*"
_class
loc:@pi/dense/kernel
�
save_30/Assign_8Assignpi/dense/kernel/Adamsave_30/RestoreV2:8*
_output_shapes

:@*"
_class
loc:@pi/dense/kernel*
use_locking(*
T0*
validate_shape(
�
save_30/Assign_9Assignpi/dense/kernel/Adam_1save_30/RestoreV2:9*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes

:@*
T0*
use_locking(
�
save_30/Assign_10Assignpi/dense_1/biassave_30/RestoreV2:10*
_output_shapes
:@*
T0*
use_locking(*"
_class
loc:@pi/dense_1/bias*
validate_shape(
�
save_30/Assign_11Assignpi/dense_1/bias/Adamsave_30/RestoreV2:11*
T0*
_output_shapes
:@*
use_locking(*
validate_shape(*"
_class
loc:@pi/dense_1/bias
�
save_30/Assign_12Assignpi/dense_1/bias/Adam_1save_30/RestoreV2:12*
_output_shapes
:@*
T0*"
_class
loc:@pi/dense_1/bias*
use_locking(*
validate_shape(
�
save_30/Assign_13Assignpi/dense_1/kernelsave_30/RestoreV2:13*
T0*
validate_shape(*
use_locking(*$
_class
loc:@pi/dense_1/kernel*
_output_shapes

:@@
�
save_30/Assign_14Assignpi/dense_1/kernel/Adamsave_30/RestoreV2:14*$
_class
loc:@pi/dense_1/kernel*
T0*
use_locking(*
validate_shape(*
_output_shapes

:@@
�
save_30/Assign_15Assignpi/dense_1/kernel/Adam_1save_30/RestoreV2:15*
use_locking(*$
_class
loc:@pi/dense_1/kernel*
T0*
validate_shape(*
_output_shapes

:@@
�
save_30/Assign_16Assignpi/dense_2/biassave_30/RestoreV2:16*
use_locking(*
validate_shape(*
T0*
_output_shapes
:*"
_class
loc:@pi/dense_2/bias
�
save_30/Assign_17Assignpi/dense_2/bias/Adamsave_30/RestoreV2:17*
T0*
use_locking(*
validate_shape(*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:
�
save_30/Assign_18Assignpi/dense_2/bias/Adam_1save_30/RestoreV2:18*
T0*
_output_shapes
:*
validate_shape(*"
_class
loc:@pi/dense_2/bias*
use_locking(
�
save_30/Assign_19Assignpi/dense_2/kernelsave_30/RestoreV2:19*
T0*
use_locking(*
_output_shapes

:@*
validate_shape(*$
_class
loc:@pi/dense_2/kernel
�
save_30/Assign_20Assignpi/dense_2/kernel/Adamsave_30/RestoreV2:20*
T0*
_output_shapes

:@*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
use_locking(
�
save_30/Assign_21Assignpi/dense_2/kernel/Adam_1save_30/RestoreV2:21*
use_locking(*
_output_shapes

:@*
T0*
validate_shape(*$
_class
loc:@pi/dense_2/kernel
�
save_30/Assign_22Assignv/dense/biassave_30/RestoreV2:22*
T0*
use_locking(*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
:@
�
save_30/Assign_23Assignv/dense/bias/Adamsave_30/RestoreV2:23*
_output_shapes
:@*
use_locking(*
validate_shape(*
_class
loc:@v/dense/bias*
T0
�
save_30/Assign_24Assignv/dense/bias/Adam_1save_30/RestoreV2:24*
T0*
_output_shapes
:@*
validate_shape(*
_class
loc:@v/dense/bias*
use_locking(
�
save_30/Assign_25Assignv/dense/kernelsave_30/RestoreV2:25*!
_class
loc:@v/dense/kernel*
T0*
use_locking(*
_output_shapes

:@*
validate_shape(
�
save_30/Assign_26Assignv/dense/kernel/Adamsave_30/RestoreV2:26*
T0*!
_class
loc:@v/dense/kernel*
validate_shape(*
use_locking(*
_output_shapes

:@
�
save_30/Assign_27Assignv/dense/kernel/Adam_1save_30/RestoreV2:27*
_output_shapes

:@*!
_class
loc:@v/dense/kernel*
use_locking(*
validate_shape(*
T0
�
save_30/Assign_28Assignv/dense_1/biassave_30/RestoreV2:28*
validate_shape(*!
_class
loc:@v/dense_1/bias*
T0*
use_locking(*
_output_shapes
:@
�
save_30/Assign_29Assignv/dense_1/bias/Adamsave_30/RestoreV2:29*
use_locking(*
_output_shapes
:@*!
_class
loc:@v/dense_1/bias*
T0*
validate_shape(
�
save_30/Assign_30Assignv/dense_1/bias/Adam_1save_30/RestoreV2:30*!
_class
loc:@v/dense_1/bias*
_output_shapes
:@*
validate_shape(*
use_locking(*
T0
�
save_30/Assign_31Assignv/dense_1/kernelsave_30/RestoreV2:31*
use_locking(*#
_class
loc:@v/dense_1/kernel*
validate_shape(*
_output_shapes

:@@*
T0
�
save_30/Assign_32Assignv/dense_1/kernel/Adamsave_30/RestoreV2:32*
use_locking(*
_output_shapes

:@@*
validate_shape(*
T0*#
_class
loc:@v/dense_1/kernel
�
save_30/Assign_33Assignv/dense_1/kernel/Adam_1save_30/RestoreV2:33*
validate_shape(*#
_class
loc:@v/dense_1/kernel*
T0*
_output_shapes

:@@*
use_locking(
�
save_30/Assign_34Assignv/dense_2/biassave_30/RestoreV2:34*
_output_shapes
:*
validate_shape(*
T0*!
_class
loc:@v/dense_2/bias*
use_locking(
�
save_30/Assign_35Assignv/dense_2/bias/Adamsave_30/RestoreV2:35*
_output_shapes
:*
validate_shape(*
use_locking(*!
_class
loc:@v/dense_2/bias*
T0
�
save_30/Assign_36Assignv/dense_2/bias/Adam_1save_30/RestoreV2:36*
use_locking(*
validate_shape(*
T0*!
_class
loc:@v/dense_2/bias*
_output_shapes
:
�
save_30/Assign_37Assignv/dense_2/kernelsave_30/RestoreV2:37*
_output_shapes

:@*
T0*#
_class
loc:@v/dense_2/kernel*
use_locking(*
validate_shape(
�
save_30/Assign_38Assignv/dense_2/kernel/Adamsave_30/RestoreV2:38*
validate_shape(*
T0*#
_class
loc:@v/dense_2/kernel*
_output_shapes

:@*
use_locking(
�
save_30/Assign_39Assignv/dense_2/kernel/Adam_1save_30/RestoreV2:39*#
_class
loc:@v/dense_2/kernel*
_output_shapes

:@*
use_locking(*
T0*
validate_shape(
�
save_30/restore_shardNoOp^save_30/Assign^save_30/Assign_1^save_30/Assign_10^save_30/Assign_11^save_30/Assign_12^save_30/Assign_13^save_30/Assign_14^save_30/Assign_15^save_30/Assign_16^save_30/Assign_17^save_30/Assign_18^save_30/Assign_19^save_30/Assign_2^save_30/Assign_20^save_30/Assign_21^save_30/Assign_22^save_30/Assign_23^save_30/Assign_24^save_30/Assign_25^save_30/Assign_26^save_30/Assign_27^save_30/Assign_28^save_30/Assign_29^save_30/Assign_3^save_30/Assign_30^save_30/Assign_31^save_30/Assign_32^save_30/Assign_33^save_30/Assign_34^save_30/Assign_35^save_30/Assign_36^save_30/Assign_37^save_30/Assign_38^save_30/Assign_39^save_30/Assign_4^save_30/Assign_5^save_30/Assign_6^save_30/Assign_7^save_30/Assign_8^save_30/Assign_9
3
save_30/restore_allNoOp^save_30/restore_shard
\
save_31/filename/inputConst*
_output_shapes
: *
valueB Bmodel*
dtype0
t
save_31/filenamePlaceholderWithDefaultsave_31/filename/input*
dtype0*
_output_shapes
: *
shape: 
k
save_31/ConstPlaceholderWithDefaultsave_31/filename*
shape: *
dtype0*
_output_shapes
: 
�
save_31/StringJoin/inputs_1Const*
_output_shapes
: *<
value3B1 B+_temp_8992d60f97ab4d4ea294bd6a9feda85a/part*
dtype0
~
save_31/StringJoin
StringJoinsave_31/Constsave_31/StringJoin/inputs_1*
	separator *
_output_shapes
: *
N
T
save_31/num_shardsConst*
value	B :*
_output_shapes
: *
dtype0
_
save_31/ShardedFilename/shardConst*
dtype0*
_output_shapes
: *
value	B : 
�
save_31/ShardedFilenameShardedFilenamesave_31/StringJoinsave_31/ShardedFilename/shardsave_31/num_shards*
_output_shapes
: 
�
save_31/SaveV2/tensor_namesConst*
_output_shapes
:(*
dtype0*�
value�B�(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1
�
save_31/SaveV2/shape_and_slicesConst*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:(
�
save_31/SaveV2SaveV2save_31/ShardedFilenamesave_31/SaveV2/tensor_namessave_31/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1v/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1v/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1v/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1v/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1v/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1v/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1*6
dtypes,
*2(
�
save_31/control_dependencyIdentitysave_31/ShardedFilename^save_31/SaveV2*
_output_shapes
: **
_class 
loc:@save_31/ShardedFilename*
T0
�
.save_31/MergeV2Checkpoints/checkpoint_prefixesPacksave_31/ShardedFilename^save_31/control_dependency*
T0*

axis *
N*
_output_shapes
:
�
save_31/MergeV2CheckpointsMergeV2Checkpoints.save_31/MergeV2Checkpoints/checkpoint_prefixessave_31/Const*
delete_old_dirs(
�
save_31/IdentityIdentitysave_31/Const^save_31/MergeV2Checkpoints^save_31/control_dependency*
T0*
_output_shapes
: 
�
save_31/RestoreV2/tensor_namesConst*
_output_shapes
:(*
dtype0*�
value�B�(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1
�
"save_31/RestoreV2/shape_and_slicesConst*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:(
�
save_31/RestoreV2	RestoreV2save_31/Constsave_31/RestoreV2/tensor_names"save_31/RestoreV2/shape_and_slices*6
dtypes,
*2(*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::
�
save_31/AssignAssignbeta1_powersave_31/RestoreV2* 
_class
loc:@pi/dense/bias*
_output_shapes
: *
use_locking(*
validate_shape(*
T0
�
save_31/Assign_1Assignbeta1_power_1save_31/RestoreV2:1*
use_locking(*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
: *
T0
�
save_31/Assign_2Assignbeta2_powersave_31/RestoreV2:2* 
_class
loc:@pi/dense/bias*
use_locking(*
T0*
validate_shape(*
_output_shapes
: 
�
save_31/Assign_3Assignbeta2_power_1save_31/RestoreV2:3*
use_locking(*
T0*
_class
loc:@v/dense/bias*
_output_shapes
: *
validate_shape(
�
save_31/Assign_4Assignpi/dense/biassave_31/RestoreV2:4*
validate_shape(*
use_locking(*
_output_shapes
:@* 
_class
loc:@pi/dense/bias*
T0
�
save_31/Assign_5Assignpi/dense/bias/Adamsave_31/RestoreV2:5*
validate_shape(*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
_output_shapes
:@
�
save_31/Assign_6Assignpi/dense/bias/Adam_1save_31/RestoreV2:6*
validate_shape(* 
_class
loc:@pi/dense/bias*
_output_shapes
:@*
T0*
use_locking(
�
save_31/Assign_7Assignpi/dense/kernelsave_31/RestoreV2:7*
validate_shape(*
T0*"
_class
loc:@pi/dense/kernel*
use_locking(*
_output_shapes

:@
�
save_31/Assign_8Assignpi/dense/kernel/Adamsave_31/RestoreV2:8*
validate_shape(*
T0*
_output_shapes

:@*"
_class
loc:@pi/dense/kernel*
use_locking(
�
save_31/Assign_9Assignpi/dense/kernel/Adam_1save_31/RestoreV2:9*"
_class
loc:@pi/dense/kernel*
T0*
_output_shapes

:@*
validate_shape(*
use_locking(
�
save_31/Assign_10Assignpi/dense_1/biassave_31/RestoreV2:10*
validate_shape(*"
_class
loc:@pi/dense_1/bias*
use_locking(*
T0*
_output_shapes
:@
�
save_31/Assign_11Assignpi/dense_1/bias/Adamsave_31/RestoreV2:11*
_output_shapes
:@*
T0*
validate_shape(*
use_locking(*"
_class
loc:@pi/dense_1/bias
�
save_31/Assign_12Assignpi/dense_1/bias/Adam_1save_31/RestoreV2:12*
use_locking(*
_output_shapes
:@*"
_class
loc:@pi/dense_1/bias*
T0*
validate_shape(
�
save_31/Assign_13Assignpi/dense_1/kernelsave_31/RestoreV2:13*
_output_shapes

:@@*
use_locking(*
validate_shape(*
T0*$
_class
loc:@pi/dense_1/kernel
�
save_31/Assign_14Assignpi/dense_1/kernel/Adamsave_31/RestoreV2:14*
use_locking(*
validate_shape(*
T0*$
_class
loc:@pi/dense_1/kernel*
_output_shapes

:@@
�
save_31/Assign_15Assignpi/dense_1/kernel/Adam_1save_31/RestoreV2:15*
use_locking(*
_output_shapes

:@@*$
_class
loc:@pi/dense_1/kernel*
T0*
validate_shape(
�
save_31/Assign_16Assignpi/dense_2/biassave_31/RestoreV2:16*
use_locking(*
_output_shapes
:*
T0*"
_class
loc:@pi/dense_2/bias*
validate_shape(
�
save_31/Assign_17Assignpi/dense_2/bias/Adamsave_31/RestoreV2:17*
_output_shapes
:*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
T0*
use_locking(
�
save_31/Assign_18Assignpi/dense_2/bias/Adam_1save_31/RestoreV2:18*
_output_shapes
:*
validate_shape(*
T0*
use_locking(*"
_class
loc:@pi/dense_2/bias
�
save_31/Assign_19Assignpi/dense_2/kernelsave_31/RestoreV2:19*
validate_shape(*
T0*$
_class
loc:@pi/dense_2/kernel*
_output_shapes

:@*
use_locking(
�
save_31/Assign_20Assignpi/dense_2/kernel/Adamsave_31/RestoreV2:20*$
_class
loc:@pi/dense_2/kernel*
_output_shapes

:@*
use_locking(*
validate_shape(*
T0
�
save_31/Assign_21Assignpi/dense_2/kernel/Adam_1save_31/RestoreV2:21*
_output_shapes

:@*
use_locking(*
validate_shape(*
T0*$
_class
loc:@pi/dense_2/kernel
�
save_31/Assign_22Assignv/dense/biassave_31/RestoreV2:22*
_class
loc:@v/dense/bias*
validate_shape(*
T0*
_output_shapes
:@*
use_locking(
�
save_31/Assign_23Assignv/dense/bias/Adamsave_31/RestoreV2:23*
_class
loc:@v/dense/bias*
_output_shapes
:@*
validate_shape(*
use_locking(*
T0
�
save_31/Assign_24Assignv/dense/bias/Adam_1save_31/RestoreV2:24*
T0*
use_locking(*
_class
loc:@v/dense/bias*
_output_shapes
:@*
validate_shape(
�
save_31/Assign_25Assignv/dense/kernelsave_31/RestoreV2:25*
T0*
validate_shape(*
_output_shapes

:@*
use_locking(*!
_class
loc:@v/dense/kernel
�
save_31/Assign_26Assignv/dense/kernel/Adamsave_31/RestoreV2:26*
use_locking(*
_output_shapes

:@*
validate_shape(*!
_class
loc:@v/dense/kernel*
T0
�
save_31/Assign_27Assignv/dense/kernel/Adam_1save_31/RestoreV2:27*
use_locking(*!
_class
loc:@v/dense/kernel*
validate_shape(*
T0*
_output_shapes

:@
�
save_31/Assign_28Assignv/dense_1/biassave_31/RestoreV2:28*
validate_shape(*
T0*
use_locking(*
_output_shapes
:@*!
_class
loc:@v/dense_1/bias
�
save_31/Assign_29Assignv/dense_1/bias/Adamsave_31/RestoreV2:29*
validate_shape(*
_output_shapes
:@*!
_class
loc:@v/dense_1/bias*
use_locking(*
T0
�
save_31/Assign_30Assignv/dense_1/bias/Adam_1save_31/RestoreV2:30*
use_locking(*
_output_shapes
:@*
T0*!
_class
loc:@v/dense_1/bias*
validate_shape(
�
save_31/Assign_31Assignv/dense_1/kernelsave_31/RestoreV2:31*
use_locking(*#
_class
loc:@v/dense_1/kernel*
_output_shapes

:@@*
validate_shape(*
T0
�
save_31/Assign_32Assignv/dense_1/kernel/Adamsave_31/RestoreV2:32*
validate_shape(*
T0*
use_locking(*#
_class
loc:@v/dense_1/kernel*
_output_shapes

:@@
�
save_31/Assign_33Assignv/dense_1/kernel/Adam_1save_31/RestoreV2:33*
T0*#
_class
loc:@v/dense_1/kernel*
use_locking(*
validate_shape(*
_output_shapes

:@@
�
save_31/Assign_34Assignv/dense_2/biassave_31/RestoreV2:34*!
_class
loc:@v/dense_2/bias*
_output_shapes
:*
validate_shape(*
T0*
use_locking(
�
save_31/Assign_35Assignv/dense_2/bias/Adamsave_31/RestoreV2:35*
_output_shapes
:*
T0*
use_locking(*
validate_shape(*!
_class
loc:@v/dense_2/bias
�
save_31/Assign_36Assignv/dense_2/bias/Adam_1save_31/RestoreV2:36*!
_class
loc:@v/dense_2/bias*
_output_shapes
:*
validate_shape(*
T0*
use_locking(
�
save_31/Assign_37Assignv/dense_2/kernelsave_31/RestoreV2:37*#
_class
loc:@v/dense_2/kernel*
_output_shapes

:@*
use_locking(*
validate_shape(*
T0
�
save_31/Assign_38Assignv/dense_2/kernel/Adamsave_31/RestoreV2:38*#
_class
loc:@v/dense_2/kernel*
use_locking(*
_output_shapes

:@*
T0*
validate_shape(
�
save_31/Assign_39Assignv/dense_2/kernel/Adam_1save_31/RestoreV2:39*#
_class
loc:@v/dense_2/kernel*
_output_shapes

:@*
T0*
use_locking(*
validate_shape(
�
save_31/restore_shardNoOp^save_31/Assign^save_31/Assign_1^save_31/Assign_10^save_31/Assign_11^save_31/Assign_12^save_31/Assign_13^save_31/Assign_14^save_31/Assign_15^save_31/Assign_16^save_31/Assign_17^save_31/Assign_18^save_31/Assign_19^save_31/Assign_2^save_31/Assign_20^save_31/Assign_21^save_31/Assign_22^save_31/Assign_23^save_31/Assign_24^save_31/Assign_25^save_31/Assign_26^save_31/Assign_27^save_31/Assign_28^save_31/Assign_29^save_31/Assign_3^save_31/Assign_30^save_31/Assign_31^save_31/Assign_32^save_31/Assign_33^save_31/Assign_34^save_31/Assign_35^save_31/Assign_36^save_31/Assign_37^save_31/Assign_38^save_31/Assign_39^save_31/Assign_4^save_31/Assign_5^save_31/Assign_6^save_31/Assign_7^save_31/Assign_8^save_31/Assign_9
3
save_31/restore_allNoOp^save_31/restore_shard
\
save_32/filename/inputConst*
dtype0*
valueB Bmodel*
_output_shapes
: 
t
save_32/filenamePlaceholderWithDefaultsave_32/filename/input*
shape: *
_output_shapes
: *
dtype0
k
save_32/ConstPlaceholderWithDefaultsave_32/filename*
dtype0*
_output_shapes
: *
shape: 
�
save_32/StringJoin/inputs_1Const*<
value3B1 B+_temp_6a0fa9b735684920a2f94c09bfa43351/part*
dtype0*
_output_shapes
: 
~
save_32/StringJoin
StringJoinsave_32/Constsave_32/StringJoin/inputs_1*
_output_shapes
: *
	separator *
N
T
save_32/num_shardsConst*
dtype0*
value	B :*
_output_shapes
: 
_
save_32/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 
�
save_32/ShardedFilenameShardedFilenamesave_32/StringJoinsave_32/ShardedFilename/shardsave_32/num_shards*
_output_shapes
: 
�
save_32/SaveV2/tensor_namesConst*
_output_shapes
:(*
dtype0*�
value�B�(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1
�
save_32/SaveV2/shape_and_slicesConst*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:(
�
save_32/SaveV2SaveV2save_32/ShardedFilenamesave_32/SaveV2/tensor_namessave_32/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1v/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1v/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1v/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1v/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1v/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1v/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1*6
dtypes,
*2(
�
save_32/control_dependencyIdentitysave_32/ShardedFilename^save_32/SaveV2**
_class 
loc:@save_32/ShardedFilename*
T0*
_output_shapes
: 
�
.save_32/MergeV2Checkpoints/checkpoint_prefixesPacksave_32/ShardedFilename^save_32/control_dependency*
T0*
N*

axis *
_output_shapes
:
�
save_32/MergeV2CheckpointsMergeV2Checkpoints.save_32/MergeV2Checkpoints/checkpoint_prefixessave_32/Const*
delete_old_dirs(
�
save_32/IdentityIdentitysave_32/Const^save_32/MergeV2Checkpoints^save_32/control_dependency*
_output_shapes
: *
T0
�
save_32/RestoreV2/tensor_namesConst*
_output_shapes
:(*�
value�B�(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
dtype0
�
"save_32/RestoreV2/shape_and_slicesConst*
_output_shapes
:(*
dtype0*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
�
save_32/RestoreV2	RestoreV2save_32/Constsave_32/RestoreV2/tensor_names"save_32/RestoreV2/shape_and_slices*6
dtypes,
*2(*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::
�
save_32/AssignAssignbeta1_powersave_32/RestoreV2*
T0*
use_locking(*
_output_shapes
: * 
_class
loc:@pi/dense/bias*
validate_shape(
�
save_32/Assign_1Assignbeta1_power_1save_32/RestoreV2:1*
use_locking(*
validate_shape(*
T0*
_output_shapes
: *
_class
loc:@v/dense/bias
�
save_32/Assign_2Assignbeta2_powersave_32/RestoreV2:2*
T0*
use_locking(* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
: 
�
save_32/Assign_3Assignbeta2_power_1save_32/RestoreV2:3*
T0*
validate_shape(*
_class
loc:@v/dense/bias*
_output_shapes
: *
use_locking(
�
save_32/Assign_4Assignpi/dense/biassave_32/RestoreV2:4*
validate_shape(* 
_class
loc:@pi/dense/bias*
T0*
use_locking(*
_output_shapes
:@
�
save_32/Assign_5Assignpi/dense/bias/Adamsave_32/RestoreV2:5*
T0*
use_locking(*
validate_shape(* 
_class
loc:@pi/dense/bias*
_output_shapes
:@
�
save_32/Assign_6Assignpi/dense/bias/Adam_1save_32/RestoreV2:6* 
_class
loc:@pi/dense/bias*
use_locking(*
_output_shapes
:@*
T0*
validate_shape(
�
save_32/Assign_7Assignpi/dense/kernelsave_32/RestoreV2:7*
_output_shapes

:@*"
_class
loc:@pi/dense/kernel*
validate_shape(*
T0*
use_locking(
�
save_32/Assign_8Assignpi/dense/kernel/Adamsave_32/RestoreV2:8*
validate_shape(*"
_class
loc:@pi/dense/kernel*
use_locking(*
_output_shapes

:@*
T0
�
save_32/Assign_9Assignpi/dense/kernel/Adam_1save_32/RestoreV2:9*"
_class
loc:@pi/dense/kernel*
_output_shapes

:@*
T0*
validate_shape(*
use_locking(
�
save_32/Assign_10Assignpi/dense_1/biassave_32/RestoreV2:10*
T0*
use_locking(*
validate_shape(*"
_class
loc:@pi/dense_1/bias*
_output_shapes
:@
�
save_32/Assign_11Assignpi/dense_1/bias/Adamsave_32/RestoreV2:11*
validate_shape(*"
_class
loc:@pi/dense_1/bias*
_output_shapes
:@*
T0*
use_locking(
�
save_32/Assign_12Assignpi/dense_1/bias/Adam_1save_32/RestoreV2:12*"
_class
loc:@pi/dense_1/bias*
_output_shapes
:@*
use_locking(*
T0*
validate_shape(
�
save_32/Assign_13Assignpi/dense_1/kernelsave_32/RestoreV2:13*
use_locking(*
_output_shapes

:@@*
T0*$
_class
loc:@pi/dense_1/kernel*
validate_shape(
�
save_32/Assign_14Assignpi/dense_1/kernel/Adamsave_32/RestoreV2:14*
validate_shape(*
T0*
use_locking(*$
_class
loc:@pi/dense_1/kernel*
_output_shapes

:@@
�
save_32/Assign_15Assignpi/dense_1/kernel/Adam_1save_32/RestoreV2:15*
validate_shape(*$
_class
loc:@pi/dense_1/kernel*
_output_shapes

:@@*
use_locking(*
T0
�
save_32/Assign_16Assignpi/dense_2/biassave_32/RestoreV2:16*
T0*
_output_shapes
:*"
_class
loc:@pi/dense_2/bias*
use_locking(*
validate_shape(
�
save_32/Assign_17Assignpi/dense_2/bias/Adamsave_32/RestoreV2:17*
_output_shapes
:*
use_locking(*
validate_shape(*"
_class
loc:@pi/dense_2/bias*
T0
�
save_32/Assign_18Assignpi/dense_2/bias/Adam_1save_32/RestoreV2:18*
use_locking(*
T0*
_output_shapes
:*
validate_shape(*"
_class
loc:@pi/dense_2/bias
�
save_32/Assign_19Assignpi/dense_2/kernelsave_32/RestoreV2:19*
_output_shapes

:@*
use_locking(*$
_class
loc:@pi/dense_2/kernel*
T0*
validate_shape(
�
save_32/Assign_20Assignpi/dense_2/kernel/Adamsave_32/RestoreV2:20*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
T0*
use_locking(*
_output_shapes

:@
�
save_32/Assign_21Assignpi/dense_2/kernel/Adam_1save_32/RestoreV2:21*
_output_shapes

:@*$
_class
loc:@pi/dense_2/kernel*
T0*
use_locking(*
validate_shape(
�
save_32/Assign_22Assignv/dense/biassave_32/RestoreV2:22*
validate_shape(*
use_locking(*
T0*
_class
loc:@v/dense/bias*
_output_shapes
:@
�
save_32/Assign_23Assignv/dense/bias/Adamsave_32/RestoreV2:23*
validate_shape(*
T0*
_class
loc:@v/dense/bias*
use_locking(*
_output_shapes
:@
�
save_32/Assign_24Assignv/dense/bias/Adam_1save_32/RestoreV2:24*
T0*
use_locking(*
_class
loc:@v/dense/bias*
_output_shapes
:@*
validate_shape(
�
save_32/Assign_25Assignv/dense/kernelsave_32/RestoreV2:25*
validate_shape(*!
_class
loc:@v/dense/kernel*
T0*
_output_shapes

:@*
use_locking(
�
save_32/Assign_26Assignv/dense/kernel/Adamsave_32/RestoreV2:26*
T0*
_output_shapes

:@*!
_class
loc:@v/dense/kernel*
validate_shape(*
use_locking(
�
save_32/Assign_27Assignv/dense/kernel/Adam_1save_32/RestoreV2:27*
use_locking(*
validate_shape(*
_output_shapes

:@*
T0*!
_class
loc:@v/dense/kernel
�
save_32/Assign_28Assignv/dense_1/biassave_32/RestoreV2:28*
use_locking(*!
_class
loc:@v/dense_1/bias*
_output_shapes
:@*
validate_shape(*
T0
�
save_32/Assign_29Assignv/dense_1/bias/Adamsave_32/RestoreV2:29*
_output_shapes
:@*
validate_shape(*
use_locking(*!
_class
loc:@v/dense_1/bias*
T0
�
save_32/Assign_30Assignv/dense_1/bias/Adam_1save_32/RestoreV2:30*
T0*!
_class
loc:@v/dense_1/bias*
_output_shapes
:@*
validate_shape(*
use_locking(
�
save_32/Assign_31Assignv/dense_1/kernelsave_32/RestoreV2:31*
validate_shape(*
T0*
use_locking(*#
_class
loc:@v/dense_1/kernel*
_output_shapes

:@@
�
save_32/Assign_32Assignv/dense_1/kernel/Adamsave_32/RestoreV2:32*
validate_shape(*#
_class
loc:@v/dense_1/kernel*
use_locking(*
_output_shapes

:@@*
T0
�
save_32/Assign_33Assignv/dense_1/kernel/Adam_1save_32/RestoreV2:33*
use_locking(*
validate_shape(*#
_class
loc:@v/dense_1/kernel*
T0*
_output_shapes

:@@
�
save_32/Assign_34Assignv/dense_2/biassave_32/RestoreV2:34*
_output_shapes
:*
validate_shape(*
T0*
use_locking(*!
_class
loc:@v/dense_2/bias
�
save_32/Assign_35Assignv/dense_2/bias/Adamsave_32/RestoreV2:35*!
_class
loc:@v/dense_2/bias*
_output_shapes
:*
T0*
validate_shape(*
use_locking(
�
save_32/Assign_36Assignv/dense_2/bias/Adam_1save_32/RestoreV2:36*
validate_shape(*
_output_shapes
:*
T0*!
_class
loc:@v/dense_2/bias*
use_locking(
�
save_32/Assign_37Assignv/dense_2/kernelsave_32/RestoreV2:37*
use_locking(*
validate_shape(*
_output_shapes

:@*#
_class
loc:@v/dense_2/kernel*
T0
�
save_32/Assign_38Assignv/dense_2/kernel/Adamsave_32/RestoreV2:38*
validate_shape(*
_output_shapes

:@*
T0*#
_class
loc:@v/dense_2/kernel*
use_locking(
�
save_32/Assign_39Assignv/dense_2/kernel/Adam_1save_32/RestoreV2:39*
use_locking(*
validate_shape(*#
_class
loc:@v/dense_2/kernel*
_output_shapes

:@*
T0
�
save_32/restore_shardNoOp^save_32/Assign^save_32/Assign_1^save_32/Assign_10^save_32/Assign_11^save_32/Assign_12^save_32/Assign_13^save_32/Assign_14^save_32/Assign_15^save_32/Assign_16^save_32/Assign_17^save_32/Assign_18^save_32/Assign_19^save_32/Assign_2^save_32/Assign_20^save_32/Assign_21^save_32/Assign_22^save_32/Assign_23^save_32/Assign_24^save_32/Assign_25^save_32/Assign_26^save_32/Assign_27^save_32/Assign_28^save_32/Assign_29^save_32/Assign_3^save_32/Assign_30^save_32/Assign_31^save_32/Assign_32^save_32/Assign_33^save_32/Assign_34^save_32/Assign_35^save_32/Assign_36^save_32/Assign_37^save_32/Assign_38^save_32/Assign_39^save_32/Assign_4^save_32/Assign_5^save_32/Assign_6^save_32/Assign_7^save_32/Assign_8^save_32/Assign_9
3
save_32/restore_allNoOp^save_32/restore_shard
\
save_33/filename/inputConst*
_output_shapes
: *
valueB Bmodel*
dtype0
t
save_33/filenamePlaceholderWithDefaultsave_33/filename/input*
shape: *
dtype0*
_output_shapes
: 
k
save_33/ConstPlaceholderWithDefaultsave_33/filename*
shape: *
_output_shapes
: *
dtype0
�
save_33/StringJoin/inputs_1Const*
dtype0*<
value3B1 B+_temp_bbefcc612f5346c89b0906e3f29b7195/part*
_output_shapes
: 
~
save_33/StringJoin
StringJoinsave_33/Constsave_33/StringJoin/inputs_1*
	separator *
_output_shapes
: *
N
T
save_33/num_shardsConst*
_output_shapes
: *
value	B :*
dtype0
_
save_33/ShardedFilename/shardConst*
dtype0*
value	B : *
_output_shapes
: 
�
save_33/ShardedFilenameShardedFilenamesave_33/StringJoinsave_33/ShardedFilename/shardsave_33/num_shards*
_output_shapes
: 
�
save_33/SaveV2/tensor_namesConst*
_output_shapes
:(*
dtype0*�
value�B�(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1
�
save_33/SaveV2/shape_and_slicesConst*
_output_shapes
:(*
dtype0*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
�
save_33/SaveV2SaveV2save_33/ShardedFilenamesave_33/SaveV2/tensor_namessave_33/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1v/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1v/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1v/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1v/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1v/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1v/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1*6
dtypes,
*2(
�
save_33/control_dependencyIdentitysave_33/ShardedFilename^save_33/SaveV2**
_class 
loc:@save_33/ShardedFilename*
T0*
_output_shapes
: 
�
.save_33/MergeV2Checkpoints/checkpoint_prefixesPacksave_33/ShardedFilename^save_33/control_dependency*
_output_shapes
:*
N*

axis *
T0
�
save_33/MergeV2CheckpointsMergeV2Checkpoints.save_33/MergeV2Checkpoints/checkpoint_prefixessave_33/Const*
delete_old_dirs(
�
save_33/IdentityIdentitysave_33/Const^save_33/MergeV2Checkpoints^save_33/control_dependency*
_output_shapes
: *
T0
�
save_33/RestoreV2/tensor_namesConst*
dtype0*
_output_shapes
:(*�
value�B�(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1
�
"save_33/RestoreV2/shape_and_slicesConst*
_output_shapes
:(*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
�
save_33/RestoreV2	RestoreV2save_33/Constsave_33/RestoreV2/tensor_names"save_33/RestoreV2/shape_and_slices*6
dtypes,
*2(*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::
�
save_33/AssignAssignbeta1_powersave_33/RestoreV2*
T0* 
_class
loc:@pi/dense/bias*
_output_shapes
: *
validate_shape(*
use_locking(
�
save_33/Assign_1Assignbeta1_power_1save_33/RestoreV2:1*
validate_shape(*
_output_shapes
: *
use_locking(*
_class
loc:@v/dense/bias*
T0
�
save_33/Assign_2Assignbeta2_powersave_33/RestoreV2:2*
validate_shape(*
T0*
_output_shapes
: * 
_class
loc:@pi/dense/bias*
use_locking(
�
save_33/Assign_3Assignbeta2_power_1save_33/RestoreV2:3*
T0*
_output_shapes
: *
_class
loc:@v/dense/bias*
validate_shape(*
use_locking(
�
save_33/Assign_4Assignpi/dense/biassave_33/RestoreV2:4*
use_locking(* 
_class
loc:@pi/dense/bias*
_output_shapes
:@*
T0*
validate_shape(
�
save_33/Assign_5Assignpi/dense/bias/Adamsave_33/RestoreV2:5*
validate_shape(* 
_class
loc:@pi/dense/bias*
use_locking(*
_output_shapes
:@*
T0
�
save_33/Assign_6Assignpi/dense/bias/Adam_1save_33/RestoreV2:6*
T0*
_output_shapes
:@*
use_locking(*
validate_shape(* 
_class
loc:@pi/dense/bias
�
save_33/Assign_7Assignpi/dense/kernelsave_33/RestoreV2:7*
validate_shape(*
use_locking(*
_output_shapes

:@*"
_class
loc:@pi/dense/kernel*
T0
�
save_33/Assign_8Assignpi/dense/kernel/Adamsave_33/RestoreV2:8*
use_locking(*"
_class
loc:@pi/dense/kernel*
validate_shape(*
T0*
_output_shapes

:@
�
save_33/Assign_9Assignpi/dense/kernel/Adam_1save_33/RestoreV2:9*
T0*"
_class
loc:@pi/dense/kernel*
use_locking(*
_output_shapes

:@*
validate_shape(
�
save_33/Assign_10Assignpi/dense_1/biassave_33/RestoreV2:10*
use_locking(*
_output_shapes
:@*
validate_shape(*
T0*"
_class
loc:@pi/dense_1/bias
�
save_33/Assign_11Assignpi/dense_1/bias/Adamsave_33/RestoreV2:11*
validate_shape(*"
_class
loc:@pi/dense_1/bias*
use_locking(*
T0*
_output_shapes
:@
�
save_33/Assign_12Assignpi/dense_1/bias/Adam_1save_33/RestoreV2:12*
_output_shapes
:@*
use_locking(*
T0*"
_class
loc:@pi/dense_1/bias*
validate_shape(
�
save_33/Assign_13Assignpi/dense_1/kernelsave_33/RestoreV2:13*$
_class
loc:@pi/dense_1/kernel*
use_locking(*
_output_shapes

:@@*
validate_shape(*
T0
�
save_33/Assign_14Assignpi/dense_1/kernel/Adamsave_33/RestoreV2:14*
_output_shapes

:@@*
validate_shape(*$
_class
loc:@pi/dense_1/kernel*
T0*
use_locking(
�
save_33/Assign_15Assignpi/dense_1/kernel/Adam_1save_33/RestoreV2:15*
_output_shapes

:@@*$
_class
loc:@pi/dense_1/kernel*
T0*
validate_shape(*
use_locking(
�
save_33/Assign_16Assignpi/dense_2/biassave_33/RestoreV2:16*
use_locking(*
validate_shape(*
T0*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:
�
save_33/Assign_17Assignpi/dense_2/bias/Adamsave_33/RestoreV2:17*
_output_shapes
:*
T0*"
_class
loc:@pi/dense_2/bias*
use_locking(*
validate_shape(
�
save_33/Assign_18Assignpi/dense_2/bias/Adam_1save_33/RestoreV2:18*
T0*
_output_shapes
:*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
use_locking(
�
save_33/Assign_19Assignpi/dense_2/kernelsave_33/RestoreV2:19*
validate_shape(*
T0*
_output_shapes

:@*
use_locking(*$
_class
loc:@pi/dense_2/kernel
�
save_33/Assign_20Assignpi/dense_2/kernel/Adamsave_33/RestoreV2:20*
use_locking(*
validate_shape(*
_output_shapes

:@*$
_class
loc:@pi/dense_2/kernel*
T0
�
save_33/Assign_21Assignpi/dense_2/kernel/Adam_1save_33/RestoreV2:21*
_output_shapes

:@*
use_locking(*
T0*$
_class
loc:@pi/dense_2/kernel*
validate_shape(
�
save_33/Assign_22Assignv/dense/biassave_33/RestoreV2:22*
validate_shape(*
T0*
_class
loc:@v/dense/bias*
_output_shapes
:@*
use_locking(
�
save_33/Assign_23Assignv/dense/bias/Adamsave_33/RestoreV2:23*
use_locking(*
validate_shape(*
_class
loc:@v/dense/bias*
_output_shapes
:@*
T0
�
save_33/Assign_24Assignv/dense/bias/Adam_1save_33/RestoreV2:24*
use_locking(*
_output_shapes
:@*
_class
loc:@v/dense/bias*
validate_shape(*
T0
�
save_33/Assign_25Assignv/dense/kernelsave_33/RestoreV2:25*
validate_shape(*
T0*
_output_shapes

:@*
use_locking(*!
_class
loc:@v/dense/kernel
�
save_33/Assign_26Assignv/dense/kernel/Adamsave_33/RestoreV2:26*
use_locking(*
T0*!
_class
loc:@v/dense/kernel*
_output_shapes

:@*
validate_shape(
�
save_33/Assign_27Assignv/dense/kernel/Adam_1save_33/RestoreV2:27*
validate_shape(*
T0*
_output_shapes

:@*!
_class
loc:@v/dense/kernel*
use_locking(
�
save_33/Assign_28Assignv/dense_1/biassave_33/RestoreV2:28*
use_locking(*!
_class
loc:@v/dense_1/bias*
_output_shapes
:@*
T0*
validate_shape(
�
save_33/Assign_29Assignv/dense_1/bias/Adamsave_33/RestoreV2:29*
T0*
validate_shape(*!
_class
loc:@v/dense_1/bias*
_output_shapes
:@*
use_locking(
�
save_33/Assign_30Assignv/dense_1/bias/Adam_1save_33/RestoreV2:30*
validate_shape(*
_output_shapes
:@*!
_class
loc:@v/dense_1/bias*
T0*
use_locking(
�
save_33/Assign_31Assignv/dense_1/kernelsave_33/RestoreV2:31*
T0*
use_locking(*
validate_shape(*
_output_shapes

:@@*#
_class
loc:@v/dense_1/kernel
�
save_33/Assign_32Assignv/dense_1/kernel/Adamsave_33/RestoreV2:32*
T0*
validate_shape(*#
_class
loc:@v/dense_1/kernel*
use_locking(*
_output_shapes

:@@
�
save_33/Assign_33Assignv/dense_1/kernel/Adam_1save_33/RestoreV2:33*
validate_shape(*
T0*
use_locking(*
_output_shapes

:@@*#
_class
loc:@v/dense_1/kernel
�
save_33/Assign_34Assignv/dense_2/biassave_33/RestoreV2:34*
T0*
validate_shape(*
_output_shapes
:*!
_class
loc:@v/dense_2/bias*
use_locking(
�
save_33/Assign_35Assignv/dense_2/bias/Adamsave_33/RestoreV2:35*
T0*
validate_shape(*
use_locking(*
_output_shapes
:*!
_class
loc:@v/dense_2/bias
�
save_33/Assign_36Assignv/dense_2/bias/Adam_1save_33/RestoreV2:36*
use_locking(*!
_class
loc:@v/dense_2/bias*
validate_shape(*
T0*
_output_shapes
:
�
save_33/Assign_37Assignv/dense_2/kernelsave_33/RestoreV2:37*#
_class
loc:@v/dense_2/kernel*
T0*
_output_shapes

:@*
validate_shape(*
use_locking(
�
save_33/Assign_38Assignv/dense_2/kernel/Adamsave_33/RestoreV2:38*
validate_shape(*
T0*
use_locking(*#
_class
loc:@v/dense_2/kernel*
_output_shapes

:@
�
save_33/Assign_39Assignv/dense_2/kernel/Adam_1save_33/RestoreV2:39*
_output_shapes

:@*
use_locking(*
validate_shape(*#
_class
loc:@v/dense_2/kernel*
T0
�
save_33/restore_shardNoOp^save_33/Assign^save_33/Assign_1^save_33/Assign_10^save_33/Assign_11^save_33/Assign_12^save_33/Assign_13^save_33/Assign_14^save_33/Assign_15^save_33/Assign_16^save_33/Assign_17^save_33/Assign_18^save_33/Assign_19^save_33/Assign_2^save_33/Assign_20^save_33/Assign_21^save_33/Assign_22^save_33/Assign_23^save_33/Assign_24^save_33/Assign_25^save_33/Assign_26^save_33/Assign_27^save_33/Assign_28^save_33/Assign_29^save_33/Assign_3^save_33/Assign_30^save_33/Assign_31^save_33/Assign_32^save_33/Assign_33^save_33/Assign_34^save_33/Assign_35^save_33/Assign_36^save_33/Assign_37^save_33/Assign_38^save_33/Assign_39^save_33/Assign_4^save_33/Assign_5^save_33/Assign_6^save_33/Assign_7^save_33/Assign_8^save_33/Assign_9
3
save_33/restore_allNoOp^save_33/restore_shard
\
save_34/filename/inputConst*
dtype0*
valueB Bmodel*
_output_shapes
: 
t
save_34/filenamePlaceholderWithDefaultsave_34/filename/input*
dtype0*
shape: *
_output_shapes
: 
k
save_34/ConstPlaceholderWithDefaultsave_34/filename*
dtype0*
shape: *
_output_shapes
: 
�
save_34/StringJoin/inputs_1Const*<
value3B1 B+_temp_7da595657bca42199aed518f1f32a470/part*
dtype0*
_output_shapes
: 
~
save_34/StringJoin
StringJoinsave_34/Constsave_34/StringJoin/inputs_1*
	separator *
_output_shapes
: *
N
T
save_34/num_shardsConst*
_output_shapes
: *
dtype0*
value	B :
_
save_34/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 
�
save_34/ShardedFilenameShardedFilenamesave_34/StringJoinsave_34/ShardedFilename/shardsave_34/num_shards*
_output_shapes
: 
�
save_34/SaveV2/tensor_namesConst*
dtype0*
_output_shapes
:(*�
value�B�(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1
�
save_34/SaveV2/shape_and_slicesConst*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:(*
dtype0
�
save_34/SaveV2SaveV2save_34/ShardedFilenamesave_34/SaveV2/tensor_namessave_34/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1v/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1v/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1v/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1v/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1v/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1v/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1*6
dtypes,
*2(
�
save_34/control_dependencyIdentitysave_34/ShardedFilename^save_34/SaveV2*
_output_shapes
: *
T0**
_class 
loc:@save_34/ShardedFilename
�
.save_34/MergeV2Checkpoints/checkpoint_prefixesPacksave_34/ShardedFilename^save_34/control_dependency*
T0*
_output_shapes
:*

axis *
N
�
save_34/MergeV2CheckpointsMergeV2Checkpoints.save_34/MergeV2Checkpoints/checkpoint_prefixessave_34/Const*
delete_old_dirs(
�
save_34/IdentityIdentitysave_34/Const^save_34/MergeV2Checkpoints^save_34/control_dependency*
_output_shapes
: *
T0
�
save_34/RestoreV2/tensor_namesConst*�
value�B�(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:(
�
"save_34/RestoreV2/shape_and_slicesConst*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:(*
dtype0
�
save_34/RestoreV2	RestoreV2save_34/Constsave_34/RestoreV2/tensor_names"save_34/RestoreV2/shape_and_slices*6
dtypes,
*2(*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::
�
save_34/AssignAssignbeta1_powersave_34/RestoreV2*
use_locking(*
validate_shape(*
T0* 
_class
loc:@pi/dense/bias*
_output_shapes
: 
�
save_34/Assign_1Assignbeta1_power_1save_34/RestoreV2:1*
_class
loc:@v/dense/bias*
use_locking(*
T0*
_output_shapes
: *
validate_shape(
�
save_34/Assign_2Assignbeta2_powersave_34/RestoreV2:2*
T0*
_output_shapes
: *
validate_shape(*
use_locking(* 
_class
loc:@pi/dense/bias
�
save_34/Assign_3Assignbeta2_power_1save_34/RestoreV2:3*
use_locking(*
validate_shape(*
_class
loc:@v/dense/bias*
T0*
_output_shapes
: 
�
save_34/Assign_4Assignpi/dense/biassave_34/RestoreV2:4*
use_locking(*
_output_shapes
:@*
validate_shape(* 
_class
loc:@pi/dense/bias*
T0
�
save_34/Assign_5Assignpi/dense/bias/Adamsave_34/RestoreV2:5*
use_locking(*
validate_shape(* 
_class
loc:@pi/dense/bias*
_output_shapes
:@*
T0
�
save_34/Assign_6Assignpi/dense/bias/Adam_1save_34/RestoreV2:6*
validate_shape(*
use_locking(*
_output_shapes
:@*
T0* 
_class
loc:@pi/dense/bias
�
save_34/Assign_7Assignpi/dense/kernelsave_34/RestoreV2:7*
T0*"
_class
loc:@pi/dense/kernel*
_output_shapes

:@*
validate_shape(*
use_locking(
�
save_34/Assign_8Assignpi/dense/kernel/Adamsave_34/RestoreV2:8*
use_locking(*
_output_shapes

:@*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(
�
save_34/Assign_9Assignpi/dense/kernel/Adam_1save_34/RestoreV2:9*
T0*
_output_shapes

:@*"
_class
loc:@pi/dense/kernel*
use_locking(*
validate_shape(
�
save_34/Assign_10Assignpi/dense_1/biassave_34/RestoreV2:10*"
_class
loc:@pi/dense_1/bias*
use_locking(*
_output_shapes
:@*
validate_shape(*
T0
�
save_34/Assign_11Assignpi/dense_1/bias/Adamsave_34/RestoreV2:11*
T0*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
use_locking(*
_output_shapes
:@
�
save_34/Assign_12Assignpi/dense_1/bias/Adam_1save_34/RestoreV2:12*"
_class
loc:@pi/dense_1/bias*
T0*
validate_shape(*
_output_shapes
:@*
use_locking(
�
save_34/Assign_13Assignpi/dense_1/kernelsave_34/RestoreV2:13*
validate_shape(*
use_locking(*
_output_shapes

:@@*$
_class
loc:@pi/dense_1/kernel*
T0
�
save_34/Assign_14Assignpi/dense_1/kernel/Adamsave_34/RestoreV2:14*$
_class
loc:@pi/dense_1/kernel*
validate_shape(*
T0*
use_locking(*
_output_shapes

:@@
�
save_34/Assign_15Assignpi/dense_1/kernel/Adam_1save_34/RestoreV2:15*
T0*
validate_shape(*$
_class
loc:@pi/dense_1/kernel*
use_locking(*
_output_shapes

:@@
�
save_34/Assign_16Assignpi/dense_2/biassave_34/RestoreV2:16*
use_locking(*
validate_shape(*
T0*
_output_shapes
:*"
_class
loc:@pi/dense_2/bias
�
save_34/Assign_17Assignpi/dense_2/bias/Adamsave_34/RestoreV2:17*
validate_shape(*
use_locking(*
_output_shapes
:*
T0*"
_class
loc:@pi/dense_2/bias
�
save_34/Assign_18Assignpi/dense_2/bias/Adam_1save_34/RestoreV2:18*
validate_shape(*
use_locking(*"
_class
loc:@pi/dense_2/bias*
T0*
_output_shapes
:
�
save_34/Assign_19Assignpi/dense_2/kernelsave_34/RestoreV2:19*
validate_shape(*
use_locking(*
T0*
_output_shapes

:@*$
_class
loc:@pi/dense_2/kernel
�
save_34/Assign_20Assignpi/dense_2/kernel/Adamsave_34/RestoreV2:20*
_output_shapes

:@*
use_locking(*$
_class
loc:@pi/dense_2/kernel*
T0*
validate_shape(
�
save_34/Assign_21Assignpi/dense_2/kernel/Adam_1save_34/RestoreV2:21*$
_class
loc:@pi/dense_2/kernel*
T0*
use_locking(*
validate_shape(*
_output_shapes

:@
�
save_34/Assign_22Assignv/dense/biassave_34/RestoreV2:22*
use_locking(*
validate_shape(*
_output_shapes
:@*
T0*
_class
loc:@v/dense/bias
�
save_34/Assign_23Assignv/dense/bias/Adamsave_34/RestoreV2:23*
_class
loc:@v/dense/bias*
use_locking(*
_output_shapes
:@*
validate_shape(*
T0
�
save_34/Assign_24Assignv/dense/bias/Adam_1save_34/RestoreV2:24*
_output_shapes
:@*
validate_shape(*
use_locking(*
_class
loc:@v/dense/bias*
T0
�
save_34/Assign_25Assignv/dense/kernelsave_34/RestoreV2:25*
T0*
validate_shape(*
_output_shapes

:@*!
_class
loc:@v/dense/kernel*
use_locking(
�
save_34/Assign_26Assignv/dense/kernel/Adamsave_34/RestoreV2:26*
_output_shapes

:@*
T0*
validate_shape(*!
_class
loc:@v/dense/kernel*
use_locking(
�
save_34/Assign_27Assignv/dense/kernel/Adam_1save_34/RestoreV2:27*!
_class
loc:@v/dense/kernel*
T0*
use_locking(*
validate_shape(*
_output_shapes

:@
�
save_34/Assign_28Assignv/dense_1/biassave_34/RestoreV2:28*
use_locking(*!
_class
loc:@v/dense_1/bias*
validate_shape(*
_output_shapes
:@*
T0
�
save_34/Assign_29Assignv/dense_1/bias/Adamsave_34/RestoreV2:29*
T0*
validate_shape(*
_output_shapes
:@*!
_class
loc:@v/dense_1/bias*
use_locking(
�
save_34/Assign_30Assignv/dense_1/bias/Adam_1save_34/RestoreV2:30*
validate_shape(*!
_class
loc:@v/dense_1/bias*
_output_shapes
:@*
use_locking(*
T0
�
save_34/Assign_31Assignv/dense_1/kernelsave_34/RestoreV2:31*
validate_shape(*#
_class
loc:@v/dense_1/kernel*
T0*
use_locking(*
_output_shapes

:@@
�
save_34/Assign_32Assignv/dense_1/kernel/Adamsave_34/RestoreV2:32*
validate_shape(*#
_class
loc:@v/dense_1/kernel*
T0*
use_locking(*
_output_shapes

:@@
�
save_34/Assign_33Assignv/dense_1/kernel/Adam_1save_34/RestoreV2:33*
validate_shape(*#
_class
loc:@v/dense_1/kernel*
_output_shapes

:@@*
T0*
use_locking(
�
save_34/Assign_34Assignv/dense_2/biassave_34/RestoreV2:34*
validate_shape(*!
_class
loc:@v/dense_2/bias*
use_locking(*
_output_shapes
:*
T0
�
save_34/Assign_35Assignv/dense_2/bias/Adamsave_34/RestoreV2:35*
_output_shapes
:*
T0*!
_class
loc:@v/dense_2/bias*
validate_shape(*
use_locking(
�
save_34/Assign_36Assignv/dense_2/bias/Adam_1save_34/RestoreV2:36*
T0*
_output_shapes
:*
use_locking(*!
_class
loc:@v/dense_2/bias*
validate_shape(
�
save_34/Assign_37Assignv/dense_2/kernelsave_34/RestoreV2:37*#
_class
loc:@v/dense_2/kernel*
validate_shape(*
_output_shapes

:@*
T0*
use_locking(
�
save_34/Assign_38Assignv/dense_2/kernel/Adamsave_34/RestoreV2:38*
validate_shape(*
T0*
use_locking(*
_output_shapes

:@*#
_class
loc:@v/dense_2/kernel
�
save_34/Assign_39Assignv/dense_2/kernel/Adam_1save_34/RestoreV2:39*
validate_shape(*#
_class
loc:@v/dense_2/kernel*
use_locking(*
T0*
_output_shapes

:@
�
save_34/restore_shardNoOp^save_34/Assign^save_34/Assign_1^save_34/Assign_10^save_34/Assign_11^save_34/Assign_12^save_34/Assign_13^save_34/Assign_14^save_34/Assign_15^save_34/Assign_16^save_34/Assign_17^save_34/Assign_18^save_34/Assign_19^save_34/Assign_2^save_34/Assign_20^save_34/Assign_21^save_34/Assign_22^save_34/Assign_23^save_34/Assign_24^save_34/Assign_25^save_34/Assign_26^save_34/Assign_27^save_34/Assign_28^save_34/Assign_29^save_34/Assign_3^save_34/Assign_30^save_34/Assign_31^save_34/Assign_32^save_34/Assign_33^save_34/Assign_34^save_34/Assign_35^save_34/Assign_36^save_34/Assign_37^save_34/Assign_38^save_34/Assign_39^save_34/Assign_4^save_34/Assign_5^save_34/Assign_6^save_34/Assign_7^save_34/Assign_8^save_34/Assign_9
3
save_34/restore_allNoOp^save_34/restore_shard
\
save_35/filename/inputConst*
_output_shapes
: *
valueB Bmodel*
dtype0
t
save_35/filenamePlaceholderWithDefaultsave_35/filename/input*
_output_shapes
: *
shape: *
dtype0
k
save_35/ConstPlaceholderWithDefaultsave_35/filename*
_output_shapes
: *
shape: *
dtype0
�
save_35/StringJoin/inputs_1Const*<
value3B1 B+_temp_edbf432ae58b4de5affb33ed9b0bdab0/part*
_output_shapes
: *
dtype0
~
save_35/StringJoin
StringJoinsave_35/Constsave_35/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
T
save_35/num_shardsConst*
dtype0*
_output_shapes
: *
value	B :
_
save_35/ShardedFilename/shardConst*
dtype0*
value	B : *
_output_shapes
: 
�
save_35/ShardedFilenameShardedFilenamesave_35/StringJoinsave_35/ShardedFilename/shardsave_35/num_shards*
_output_shapes
: 
�
save_35/SaveV2/tensor_namesConst*
dtype0*
_output_shapes
:(*�
value�B�(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1
�
save_35/SaveV2/shape_and_slicesConst*
_output_shapes
:(*
dtype0*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
�
save_35/SaveV2SaveV2save_35/ShardedFilenamesave_35/SaveV2/tensor_namessave_35/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1v/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1v/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1v/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1v/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1v/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1v/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1*6
dtypes,
*2(
�
save_35/control_dependencyIdentitysave_35/ShardedFilename^save_35/SaveV2*
T0**
_class 
loc:@save_35/ShardedFilename*
_output_shapes
: 
�
.save_35/MergeV2Checkpoints/checkpoint_prefixesPacksave_35/ShardedFilename^save_35/control_dependency*
_output_shapes
:*

axis *
N*
T0
�
save_35/MergeV2CheckpointsMergeV2Checkpoints.save_35/MergeV2Checkpoints/checkpoint_prefixessave_35/Const*
delete_old_dirs(
�
save_35/IdentityIdentitysave_35/Const^save_35/MergeV2Checkpoints^save_35/control_dependency*
T0*
_output_shapes
: 
�
save_35/RestoreV2/tensor_namesConst*�
value�B�(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
_output_shapes
:(*
dtype0
�
"save_35/RestoreV2/shape_and_slicesConst*
_output_shapes
:(*
dtype0*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
�
save_35/RestoreV2	RestoreV2save_35/Constsave_35/RestoreV2/tensor_names"save_35/RestoreV2/shape_and_slices*6
dtypes,
*2(*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::
�
save_35/AssignAssignbeta1_powersave_35/RestoreV2*
use_locking(*
validate_shape(*
_output_shapes
: * 
_class
loc:@pi/dense/bias*
T0
�
save_35/Assign_1Assignbeta1_power_1save_35/RestoreV2:1*
use_locking(*
T0*
validate_shape(*
_class
loc:@v/dense/bias*
_output_shapes
: 
�
save_35/Assign_2Assignbeta2_powersave_35/RestoreV2:2*
use_locking(*
T0*
_output_shapes
: * 
_class
loc:@pi/dense/bias*
validate_shape(
�
save_35/Assign_3Assignbeta2_power_1save_35/RestoreV2:3*
_class
loc:@v/dense/bias*
T0*
use_locking(*
validate_shape(*
_output_shapes
: 
�
save_35/Assign_4Assignpi/dense/biassave_35/RestoreV2:4*
T0* 
_class
loc:@pi/dense/bias*
use_locking(*
validate_shape(*
_output_shapes
:@
�
save_35/Assign_5Assignpi/dense/bias/Adamsave_35/RestoreV2:5* 
_class
loc:@pi/dense/bias*
use_locking(*
_output_shapes
:@*
T0*
validate_shape(
�
save_35/Assign_6Assignpi/dense/bias/Adam_1save_35/RestoreV2:6*
_output_shapes
:@*
validate_shape(*
T0* 
_class
loc:@pi/dense/bias*
use_locking(
�
save_35/Assign_7Assignpi/dense/kernelsave_35/RestoreV2:7*"
_class
loc:@pi/dense/kernel*
use_locking(*
validate_shape(*
T0*
_output_shapes

:@
�
save_35/Assign_8Assignpi/dense/kernel/Adamsave_35/RestoreV2:8*
use_locking(*
_output_shapes

:@*
validate_shape(*
T0*"
_class
loc:@pi/dense/kernel
�
save_35/Assign_9Assignpi/dense/kernel/Adam_1save_35/RestoreV2:9*
use_locking(*"
_class
loc:@pi/dense/kernel*
validate_shape(*
T0*
_output_shapes

:@
�
save_35/Assign_10Assignpi/dense_1/biassave_35/RestoreV2:10*
validate_shape(*
use_locking(*
T0*
_output_shapes
:@*"
_class
loc:@pi/dense_1/bias
�
save_35/Assign_11Assignpi/dense_1/bias/Adamsave_35/RestoreV2:11*
validate_shape(*
_output_shapes
:@*
use_locking(*"
_class
loc:@pi/dense_1/bias*
T0
�
save_35/Assign_12Assignpi/dense_1/bias/Adam_1save_35/RestoreV2:12*
validate_shape(*
T0*
use_locking(*
_output_shapes
:@*"
_class
loc:@pi/dense_1/bias
�
save_35/Assign_13Assignpi/dense_1/kernelsave_35/RestoreV2:13*
_output_shapes

:@@*
T0*
use_locking(*$
_class
loc:@pi/dense_1/kernel*
validate_shape(
�
save_35/Assign_14Assignpi/dense_1/kernel/Adamsave_35/RestoreV2:14*$
_class
loc:@pi/dense_1/kernel*
_output_shapes

:@@*
use_locking(*
T0*
validate_shape(
�
save_35/Assign_15Assignpi/dense_1/kernel/Adam_1save_35/RestoreV2:15*$
_class
loc:@pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@*
use_locking(*
T0
�
save_35/Assign_16Assignpi/dense_2/biassave_35/RestoreV2:16*
use_locking(*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:*
validate_shape(*
T0
�
save_35/Assign_17Assignpi/dense_2/bias/Adamsave_35/RestoreV2:17*
use_locking(*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:*
T0*
validate_shape(
�
save_35/Assign_18Assignpi/dense_2/bias/Adam_1save_35/RestoreV2:18*
validate_shape(*
use_locking(*
_output_shapes
:*
T0*"
_class
loc:@pi/dense_2/bias
�
save_35/Assign_19Assignpi/dense_2/kernelsave_35/RestoreV2:19*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
T0*
_output_shapes

:@*
use_locking(
�
save_35/Assign_20Assignpi/dense_2/kernel/Adamsave_35/RestoreV2:20*$
_class
loc:@pi/dense_2/kernel*
use_locking(*
T0*
_output_shapes

:@*
validate_shape(
�
save_35/Assign_21Assignpi/dense_2/kernel/Adam_1save_35/RestoreV2:21*
use_locking(*
validate_shape(*
_output_shapes

:@*
T0*$
_class
loc:@pi/dense_2/kernel
�
save_35/Assign_22Assignv/dense/biassave_35/RestoreV2:22*
use_locking(*
validate_shape(*
T0*
_class
loc:@v/dense/bias*
_output_shapes
:@
�
save_35/Assign_23Assignv/dense/bias/Adamsave_35/RestoreV2:23*
_class
loc:@v/dense/bias*
T0*
validate_shape(*
_output_shapes
:@*
use_locking(
�
save_35/Assign_24Assignv/dense/bias/Adam_1save_35/RestoreV2:24*
_output_shapes
:@*
T0*
use_locking(*
_class
loc:@v/dense/bias*
validate_shape(
�
save_35/Assign_25Assignv/dense/kernelsave_35/RestoreV2:25*
T0*
validate_shape(*
use_locking(*!
_class
loc:@v/dense/kernel*
_output_shapes

:@
�
save_35/Assign_26Assignv/dense/kernel/Adamsave_35/RestoreV2:26*
validate_shape(*
use_locking(*
_output_shapes

:@*
T0*!
_class
loc:@v/dense/kernel
�
save_35/Assign_27Assignv/dense/kernel/Adam_1save_35/RestoreV2:27*
_output_shapes

:@*
use_locking(*
validate_shape(*
T0*!
_class
loc:@v/dense/kernel
�
save_35/Assign_28Assignv/dense_1/biassave_35/RestoreV2:28*
_output_shapes
:@*
validate_shape(*
use_locking(*!
_class
loc:@v/dense_1/bias*
T0
�
save_35/Assign_29Assignv/dense_1/bias/Adamsave_35/RestoreV2:29*
_output_shapes
:@*
use_locking(*!
_class
loc:@v/dense_1/bias*
validate_shape(*
T0
�
save_35/Assign_30Assignv/dense_1/bias/Adam_1save_35/RestoreV2:30*
validate_shape(*
use_locking(*!
_class
loc:@v/dense_1/bias*
T0*
_output_shapes
:@
�
save_35/Assign_31Assignv/dense_1/kernelsave_35/RestoreV2:31*
T0*
use_locking(*
validate_shape(*
_output_shapes

:@@*#
_class
loc:@v/dense_1/kernel
�
save_35/Assign_32Assignv/dense_1/kernel/Adamsave_35/RestoreV2:32*
validate_shape(*
use_locking(*
_output_shapes

:@@*
T0*#
_class
loc:@v/dense_1/kernel
�
save_35/Assign_33Assignv/dense_1/kernel/Adam_1save_35/RestoreV2:33*
T0*
validate_shape(*
_output_shapes

:@@*#
_class
loc:@v/dense_1/kernel*
use_locking(
�
save_35/Assign_34Assignv/dense_2/biassave_35/RestoreV2:34*!
_class
loc:@v/dense_2/bias*
use_locking(*
_output_shapes
:*
T0*
validate_shape(
�
save_35/Assign_35Assignv/dense_2/bias/Adamsave_35/RestoreV2:35*
use_locking(*
T0*!
_class
loc:@v/dense_2/bias*
_output_shapes
:*
validate_shape(
�
save_35/Assign_36Assignv/dense_2/bias/Adam_1save_35/RestoreV2:36*
use_locking(*
T0*!
_class
loc:@v/dense_2/bias*
validate_shape(*
_output_shapes
:
�
save_35/Assign_37Assignv/dense_2/kernelsave_35/RestoreV2:37*
T0*#
_class
loc:@v/dense_2/kernel*
use_locking(*
_output_shapes

:@*
validate_shape(
�
save_35/Assign_38Assignv/dense_2/kernel/Adamsave_35/RestoreV2:38*
validate_shape(*
T0*
_output_shapes

:@*
use_locking(*#
_class
loc:@v/dense_2/kernel
�
save_35/Assign_39Assignv/dense_2/kernel/Adam_1save_35/RestoreV2:39*
use_locking(*
_output_shapes

:@*#
_class
loc:@v/dense_2/kernel*
T0*
validate_shape(
�
save_35/restore_shardNoOp^save_35/Assign^save_35/Assign_1^save_35/Assign_10^save_35/Assign_11^save_35/Assign_12^save_35/Assign_13^save_35/Assign_14^save_35/Assign_15^save_35/Assign_16^save_35/Assign_17^save_35/Assign_18^save_35/Assign_19^save_35/Assign_2^save_35/Assign_20^save_35/Assign_21^save_35/Assign_22^save_35/Assign_23^save_35/Assign_24^save_35/Assign_25^save_35/Assign_26^save_35/Assign_27^save_35/Assign_28^save_35/Assign_29^save_35/Assign_3^save_35/Assign_30^save_35/Assign_31^save_35/Assign_32^save_35/Assign_33^save_35/Assign_34^save_35/Assign_35^save_35/Assign_36^save_35/Assign_37^save_35/Assign_38^save_35/Assign_39^save_35/Assign_4^save_35/Assign_5^save_35/Assign_6^save_35/Assign_7^save_35/Assign_8^save_35/Assign_9
3
save_35/restore_allNoOp^save_35/restore_shard
\
save_36/filename/inputConst*
dtype0*
_output_shapes
: *
valueB Bmodel
t
save_36/filenamePlaceholderWithDefaultsave_36/filename/input*
_output_shapes
: *
dtype0*
shape: 
k
save_36/ConstPlaceholderWithDefaultsave_36/filename*
dtype0*
_output_shapes
: *
shape: 
�
save_36/StringJoin/inputs_1Const*
_output_shapes
: *<
value3B1 B+_temp_ac20d8bf711d433b82cda7cfba08371b/part*
dtype0
~
save_36/StringJoin
StringJoinsave_36/Constsave_36/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
T
save_36/num_shardsConst*
_output_shapes
: *
value	B :*
dtype0
_
save_36/ShardedFilename/shardConst*
dtype0*
value	B : *
_output_shapes
: 
�
save_36/ShardedFilenameShardedFilenamesave_36/StringJoinsave_36/ShardedFilename/shardsave_36/num_shards*
_output_shapes
: 
�
save_36/SaveV2/tensor_namesConst*�
value�B�(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
_output_shapes
:(*
dtype0
�
save_36/SaveV2/shape_and_slicesConst*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:(
�
save_36/SaveV2SaveV2save_36/ShardedFilenamesave_36/SaveV2/tensor_namessave_36/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1v/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1v/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1v/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1v/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1v/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1v/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1*6
dtypes,
*2(
�
save_36/control_dependencyIdentitysave_36/ShardedFilename^save_36/SaveV2*
_output_shapes
: *
T0**
_class 
loc:@save_36/ShardedFilename
�
.save_36/MergeV2Checkpoints/checkpoint_prefixesPacksave_36/ShardedFilename^save_36/control_dependency*
N*
T0*

axis *
_output_shapes
:
�
save_36/MergeV2CheckpointsMergeV2Checkpoints.save_36/MergeV2Checkpoints/checkpoint_prefixessave_36/Const*
delete_old_dirs(
�
save_36/IdentityIdentitysave_36/Const^save_36/MergeV2Checkpoints^save_36/control_dependency*
T0*
_output_shapes
: 
�
save_36/RestoreV2/tensor_namesConst*
dtype0*
_output_shapes
:(*�
value�B�(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1
�
"save_36/RestoreV2/shape_and_slicesConst*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:(*
dtype0
�
save_36/RestoreV2	RestoreV2save_36/Constsave_36/RestoreV2/tensor_names"save_36/RestoreV2/shape_and_slices*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::*6
dtypes,
*2(
�
save_36/AssignAssignbeta1_powersave_36/RestoreV2*
T0*
validate_shape(*
_output_shapes
: *
use_locking(* 
_class
loc:@pi/dense/bias
�
save_36/Assign_1Assignbeta1_power_1save_36/RestoreV2:1*
T0*
_output_shapes
: *
_class
loc:@v/dense/bias*
use_locking(*
validate_shape(
�
save_36/Assign_2Assignbeta2_powersave_36/RestoreV2:2*
validate_shape(*
_output_shapes
: *
T0*
use_locking(* 
_class
loc:@pi/dense/bias
�
save_36/Assign_3Assignbeta2_power_1save_36/RestoreV2:3*
_output_shapes
: *
use_locking(*
T0*
validate_shape(*
_class
loc:@v/dense/bias
�
save_36/Assign_4Assignpi/dense/biassave_36/RestoreV2:4*
T0*
validate_shape(* 
_class
loc:@pi/dense/bias*
_output_shapes
:@*
use_locking(
�
save_36/Assign_5Assignpi/dense/bias/Adamsave_36/RestoreV2:5*
validate_shape(*
_output_shapes
:@*
T0* 
_class
loc:@pi/dense/bias*
use_locking(
�
save_36/Assign_6Assignpi/dense/bias/Adam_1save_36/RestoreV2:6* 
_class
loc:@pi/dense/bias*
_output_shapes
:@*
use_locking(*
validate_shape(*
T0
�
save_36/Assign_7Assignpi/dense/kernelsave_36/RestoreV2:7*
use_locking(*
_output_shapes

:@*"
_class
loc:@pi/dense/kernel*
validate_shape(*
T0
�
save_36/Assign_8Assignpi/dense/kernel/Adamsave_36/RestoreV2:8*
T0*
validate_shape(*
_output_shapes

:@*
use_locking(*"
_class
loc:@pi/dense/kernel
�
save_36/Assign_9Assignpi/dense/kernel/Adam_1save_36/RestoreV2:9*
T0*
_output_shapes

:@*
use_locking(*
validate_shape(*"
_class
loc:@pi/dense/kernel
�
save_36/Assign_10Assignpi/dense_1/biassave_36/RestoreV2:10*
_output_shapes
:@*"
_class
loc:@pi/dense_1/bias*
T0*
use_locking(*
validate_shape(
�
save_36/Assign_11Assignpi/dense_1/bias/Adamsave_36/RestoreV2:11*"
_class
loc:@pi/dense_1/bias*
T0*
validate_shape(*
use_locking(*
_output_shapes
:@
�
save_36/Assign_12Assignpi/dense_1/bias/Adam_1save_36/RestoreV2:12*
T0*
use_locking(*
_output_shapes
:@*
validate_shape(*"
_class
loc:@pi/dense_1/bias
�
save_36/Assign_13Assignpi/dense_1/kernelsave_36/RestoreV2:13*
use_locking(*$
_class
loc:@pi/dense_1/kernel*
_output_shapes

:@@*
T0*
validate_shape(
�
save_36/Assign_14Assignpi/dense_1/kernel/Adamsave_36/RestoreV2:14*
use_locking(*
validate_shape(*
T0*$
_class
loc:@pi/dense_1/kernel*
_output_shapes

:@@
�
save_36/Assign_15Assignpi/dense_1/kernel/Adam_1save_36/RestoreV2:15*$
_class
loc:@pi/dense_1/kernel*
_output_shapes

:@@*
validate_shape(*
use_locking(*
T0
�
save_36/Assign_16Assignpi/dense_2/biassave_36/RestoreV2:16*
validate_shape(*
_output_shapes
:*
T0*"
_class
loc:@pi/dense_2/bias*
use_locking(
�
save_36/Assign_17Assignpi/dense_2/bias/Adamsave_36/RestoreV2:17*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:*
validate_shape(
�
save_36/Assign_18Assignpi/dense_2/bias/Adam_1save_36/RestoreV2:18*
validate_shape(*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:*
T0*
use_locking(
�
save_36/Assign_19Assignpi/dense_2/kernelsave_36/RestoreV2:19*
_output_shapes

:@*$
_class
loc:@pi/dense_2/kernel*
T0*
use_locking(*
validate_shape(
�
save_36/Assign_20Assignpi/dense_2/kernel/Adamsave_36/RestoreV2:20*
validate_shape(*
_output_shapes

:@*$
_class
loc:@pi/dense_2/kernel*
T0*
use_locking(
�
save_36/Assign_21Assignpi/dense_2/kernel/Adam_1save_36/RestoreV2:21*
T0*
validate_shape(*
_output_shapes

:@*
use_locking(*$
_class
loc:@pi/dense_2/kernel
�
save_36/Assign_22Assignv/dense/biassave_36/RestoreV2:22*
_class
loc:@v/dense/bias*
_output_shapes
:@*
validate_shape(*
use_locking(*
T0
�
save_36/Assign_23Assignv/dense/bias/Adamsave_36/RestoreV2:23*
validate_shape(*
T0*
_class
loc:@v/dense/bias*
use_locking(*
_output_shapes
:@
�
save_36/Assign_24Assignv/dense/bias/Adam_1save_36/RestoreV2:24*
_output_shapes
:@*
use_locking(*
T0*
validate_shape(*
_class
loc:@v/dense/bias
�
save_36/Assign_25Assignv/dense/kernelsave_36/RestoreV2:25*!
_class
loc:@v/dense/kernel*
validate_shape(*
use_locking(*
T0*
_output_shapes

:@
�
save_36/Assign_26Assignv/dense/kernel/Adamsave_36/RestoreV2:26*
_output_shapes

:@*
use_locking(*
validate_shape(*!
_class
loc:@v/dense/kernel*
T0
�
save_36/Assign_27Assignv/dense/kernel/Adam_1save_36/RestoreV2:27*
_output_shapes

:@*
T0*!
_class
loc:@v/dense/kernel*
use_locking(*
validate_shape(
�
save_36/Assign_28Assignv/dense_1/biassave_36/RestoreV2:28*
T0*
validate_shape(*
_output_shapes
:@*
use_locking(*!
_class
loc:@v/dense_1/bias
�
save_36/Assign_29Assignv/dense_1/bias/Adamsave_36/RestoreV2:29*
_output_shapes
:@*
use_locking(*
T0*!
_class
loc:@v/dense_1/bias*
validate_shape(
�
save_36/Assign_30Assignv/dense_1/bias/Adam_1save_36/RestoreV2:30*
use_locking(*!
_class
loc:@v/dense_1/bias*
T0*
validate_shape(*
_output_shapes
:@
�
save_36/Assign_31Assignv/dense_1/kernelsave_36/RestoreV2:31*
T0*
_output_shapes

:@@*#
_class
loc:@v/dense_1/kernel*
use_locking(*
validate_shape(
�
save_36/Assign_32Assignv/dense_1/kernel/Adamsave_36/RestoreV2:32*
use_locking(*
validate_shape(*
_output_shapes

:@@*#
_class
loc:@v/dense_1/kernel*
T0
�
save_36/Assign_33Assignv/dense_1/kernel/Adam_1save_36/RestoreV2:33*
T0*
use_locking(*#
_class
loc:@v/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
�
save_36/Assign_34Assignv/dense_2/biassave_36/RestoreV2:34*
use_locking(*!
_class
loc:@v/dense_2/bias*
validate_shape(*
T0*
_output_shapes
:
�
save_36/Assign_35Assignv/dense_2/bias/Adamsave_36/RestoreV2:35*!
_class
loc:@v/dense_2/bias*
use_locking(*
T0*
validate_shape(*
_output_shapes
:
�
save_36/Assign_36Assignv/dense_2/bias/Adam_1save_36/RestoreV2:36*
use_locking(*
T0*
validate_shape(*
_output_shapes
:*!
_class
loc:@v/dense_2/bias
�
save_36/Assign_37Assignv/dense_2/kernelsave_36/RestoreV2:37*
_output_shapes

:@*
validate_shape(*
use_locking(*#
_class
loc:@v/dense_2/kernel*
T0
�
save_36/Assign_38Assignv/dense_2/kernel/Adamsave_36/RestoreV2:38*
validate_shape(*#
_class
loc:@v/dense_2/kernel*
T0*
_output_shapes

:@*
use_locking(
�
save_36/Assign_39Assignv/dense_2/kernel/Adam_1save_36/RestoreV2:39*
T0*#
_class
loc:@v/dense_2/kernel*
_output_shapes

:@*
validate_shape(*
use_locking(
�
save_36/restore_shardNoOp^save_36/Assign^save_36/Assign_1^save_36/Assign_10^save_36/Assign_11^save_36/Assign_12^save_36/Assign_13^save_36/Assign_14^save_36/Assign_15^save_36/Assign_16^save_36/Assign_17^save_36/Assign_18^save_36/Assign_19^save_36/Assign_2^save_36/Assign_20^save_36/Assign_21^save_36/Assign_22^save_36/Assign_23^save_36/Assign_24^save_36/Assign_25^save_36/Assign_26^save_36/Assign_27^save_36/Assign_28^save_36/Assign_29^save_36/Assign_3^save_36/Assign_30^save_36/Assign_31^save_36/Assign_32^save_36/Assign_33^save_36/Assign_34^save_36/Assign_35^save_36/Assign_36^save_36/Assign_37^save_36/Assign_38^save_36/Assign_39^save_36/Assign_4^save_36/Assign_5^save_36/Assign_6^save_36/Assign_7^save_36/Assign_8^save_36/Assign_9
3
save_36/restore_allNoOp^save_36/restore_shard
\
save_37/filename/inputConst*
_output_shapes
: *
dtype0*
valueB Bmodel
t
save_37/filenamePlaceholderWithDefaultsave_37/filename/input*
_output_shapes
: *
shape: *
dtype0
k
save_37/ConstPlaceholderWithDefaultsave_37/filename*
_output_shapes
: *
shape: *
dtype0
�
save_37/StringJoin/inputs_1Const*<
value3B1 B+_temp_f0c4a4738e284974a2bd9c9e529996c7/part*
_output_shapes
: *
dtype0
~
save_37/StringJoin
StringJoinsave_37/Constsave_37/StringJoin/inputs_1*
	separator *
_output_shapes
: *
N
T
save_37/num_shardsConst*
dtype0*
value	B :*
_output_shapes
: 
_
save_37/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 
�
save_37/ShardedFilenameShardedFilenamesave_37/StringJoinsave_37/ShardedFilename/shardsave_37/num_shards*
_output_shapes
: 
�
save_37/SaveV2/tensor_namesConst*
_output_shapes
:(*
dtype0*�
value�B�(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1
�
save_37/SaveV2/shape_and_slicesConst*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:(*
dtype0
�
save_37/SaveV2SaveV2save_37/ShardedFilenamesave_37/SaveV2/tensor_namessave_37/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1v/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1v/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1v/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1v/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1v/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1v/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1*6
dtypes,
*2(
�
save_37/control_dependencyIdentitysave_37/ShardedFilename^save_37/SaveV2*
T0**
_class 
loc:@save_37/ShardedFilename*
_output_shapes
: 
�
.save_37/MergeV2Checkpoints/checkpoint_prefixesPacksave_37/ShardedFilename^save_37/control_dependency*

axis *
_output_shapes
:*
T0*
N
�
save_37/MergeV2CheckpointsMergeV2Checkpoints.save_37/MergeV2Checkpoints/checkpoint_prefixessave_37/Const*
delete_old_dirs(
�
save_37/IdentityIdentitysave_37/Const^save_37/MergeV2Checkpoints^save_37/control_dependency*
_output_shapes
: *
T0
�
save_37/RestoreV2/tensor_namesConst*
_output_shapes
:(*�
value�B�(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
dtype0
�
"save_37/RestoreV2/shape_and_slicesConst*
_output_shapes
:(*
dtype0*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
�
save_37/RestoreV2	RestoreV2save_37/Constsave_37/RestoreV2/tensor_names"save_37/RestoreV2/shape_and_slices*6
dtypes,
*2(*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::
�
save_37/AssignAssignbeta1_powersave_37/RestoreV2* 
_class
loc:@pi/dense/bias*
_output_shapes
: *
use_locking(*
validate_shape(*
T0
�
save_37/Assign_1Assignbeta1_power_1save_37/RestoreV2:1*
validate_shape(*
T0*
_output_shapes
: *
use_locking(*
_class
loc:@v/dense/bias
�
save_37/Assign_2Assignbeta2_powersave_37/RestoreV2:2*
_output_shapes
: * 
_class
loc:@pi/dense/bias*
validate_shape(*
T0*
use_locking(
�
save_37/Assign_3Assignbeta2_power_1save_37/RestoreV2:3*
_output_shapes
: *
_class
loc:@v/dense/bias*
validate_shape(*
use_locking(*
T0
�
save_37/Assign_4Assignpi/dense/biassave_37/RestoreV2:4*
validate_shape(* 
_class
loc:@pi/dense/bias*
_output_shapes
:@*
T0*
use_locking(
�
save_37/Assign_5Assignpi/dense/bias/Adamsave_37/RestoreV2:5* 
_class
loc:@pi/dense/bias*
_output_shapes
:@*
validate_shape(*
T0*
use_locking(
�
save_37/Assign_6Assignpi/dense/bias/Adam_1save_37/RestoreV2:6*
validate_shape(*
T0*
use_locking(*
_output_shapes
:@* 
_class
loc:@pi/dense/bias
�
save_37/Assign_7Assignpi/dense/kernelsave_37/RestoreV2:7*"
_class
loc:@pi/dense/kernel*
T0*
_output_shapes

:@*
validate_shape(*
use_locking(
�
save_37/Assign_8Assignpi/dense/kernel/Adamsave_37/RestoreV2:8*"
_class
loc:@pi/dense/kernel*
_output_shapes

:@*
T0*
use_locking(*
validate_shape(
�
save_37/Assign_9Assignpi/dense/kernel/Adam_1save_37/RestoreV2:9*
use_locking(*"
_class
loc:@pi/dense/kernel*
T0*
validate_shape(*
_output_shapes

:@
�
save_37/Assign_10Assignpi/dense_1/biassave_37/RestoreV2:10*
T0*"
_class
loc:@pi/dense_1/bias*
_output_shapes
:@*
use_locking(*
validate_shape(
�
save_37/Assign_11Assignpi/dense_1/bias/Adamsave_37/RestoreV2:11*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
use_locking(*
T0*
_output_shapes
:@
�
save_37/Assign_12Assignpi/dense_1/bias/Adam_1save_37/RestoreV2:12*
T0*
validate_shape(*
use_locking(*
_output_shapes
:@*"
_class
loc:@pi/dense_1/bias
�
save_37/Assign_13Assignpi/dense_1/kernelsave_37/RestoreV2:13*
T0*
use_locking(*
_output_shapes

:@@*$
_class
loc:@pi/dense_1/kernel*
validate_shape(
�
save_37/Assign_14Assignpi/dense_1/kernel/Adamsave_37/RestoreV2:14*$
_class
loc:@pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@*
use_locking(*
T0
�
save_37/Assign_15Assignpi/dense_1/kernel/Adam_1save_37/RestoreV2:15*
validate_shape(*
use_locking(*
_output_shapes

:@@*
T0*$
_class
loc:@pi/dense_1/kernel
�
save_37/Assign_16Assignpi/dense_2/biassave_37/RestoreV2:16*
T0*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
use_locking(*
_output_shapes
:
�
save_37/Assign_17Assignpi/dense_2/bias/Adamsave_37/RestoreV2:17*
T0*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
use_locking(*
_output_shapes
:
�
save_37/Assign_18Assignpi/dense_2/bias/Adam_1save_37/RestoreV2:18*
use_locking(*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
_output_shapes
:*
T0
�
save_37/Assign_19Assignpi/dense_2/kernelsave_37/RestoreV2:19*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0*$
_class
loc:@pi/dense_2/kernel
�
save_37/Assign_20Assignpi/dense_2/kernel/Adamsave_37/RestoreV2:20*
validate_shape(*$
_class
loc:@pi/dense_2/kernel*
T0*
use_locking(*
_output_shapes

:@
�
save_37/Assign_21Assignpi/dense_2/kernel/Adam_1save_37/RestoreV2:21*
T0*
use_locking(*$
_class
loc:@pi/dense_2/kernel*
_output_shapes

:@*
validate_shape(
�
save_37/Assign_22Assignv/dense/biassave_37/RestoreV2:22*
use_locking(*
T0*
validate_shape(*
_class
loc:@v/dense/bias*
_output_shapes
:@
�
save_37/Assign_23Assignv/dense/bias/Adamsave_37/RestoreV2:23*
_class
loc:@v/dense/bias*
T0*
validate_shape(*
_output_shapes
:@*
use_locking(
�
save_37/Assign_24Assignv/dense/bias/Adam_1save_37/RestoreV2:24*
_output_shapes
:@*
_class
loc:@v/dense/bias*
validate_shape(*
T0*
use_locking(
�
save_37/Assign_25Assignv/dense/kernelsave_37/RestoreV2:25*
T0*
_output_shapes

:@*
use_locking(*!
_class
loc:@v/dense/kernel*
validate_shape(
�
save_37/Assign_26Assignv/dense/kernel/Adamsave_37/RestoreV2:26*
use_locking(*
validate_shape(*!
_class
loc:@v/dense/kernel*
_output_shapes

:@*
T0
�
save_37/Assign_27Assignv/dense/kernel/Adam_1save_37/RestoreV2:27*
use_locking(*
_output_shapes

:@*
T0*
validate_shape(*!
_class
loc:@v/dense/kernel
�
save_37/Assign_28Assignv/dense_1/biassave_37/RestoreV2:28*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*!
_class
loc:@v/dense_1/bias
�
save_37/Assign_29Assignv/dense_1/bias/Adamsave_37/RestoreV2:29*
validate_shape(*
T0*!
_class
loc:@v/dense_1/bias*
_output_shapes
:@*
use_locking(
�
save_37/Assign_30Assignv/dense_1/bias/Adam_1save_37/RestoreV2:30*
validate_shape(*
T0*!
_class
loc:@v/dense_1/bias*
use_locking(*
_output_shapes
:@
�
save_37/Assign_31Assignv/dense_1/kernelsave_37/RestoreV2:31*
use_locking(*#
_class
loc:@v/dense_1/kernel*
_output_shapes

:@@*
T0*
validate_shape(
�
save_37/Assign_32Assignv/dense_1/kernel/Adamsave_37/RestoreV2:32*
validate_shape(*
T0*
_output_shapes

:@@*
use_locking(*#
_class
loc:@v/dense_1/kernel
�
save_37/Assign_33Assignv/dense_1/kernel/Adam_1save_37/RestoreV2:33*
T0*
use_locking(*
validate_shape(*
_output_shapes

:@@*#
_class
loc:@v/dense_1/kernel
�
save_37/Assign_34Assignv/dense_2/biassave_37/RestoreV2:34*
T0*!
_class
loc:@v/dense_2/bias*
_output_shapes
:*
validate_shape(*
use_locking(
�
save_37/Assign_35Assignv/dense_2/bias/Adamsave_37/RestoreV2:35*
_output_shapes
:*
use_locking(*
validate_shape(*!
_class
loc:@v/dense_2/bias*
T0
�
save_37/Assign_36Assignv/dense_2/bias/Adam_1save_37/RestoreV2:36*
T0*
_output_shapes
:*!
_class
loc:@v/dense_2/bias*
use_locking(*
validate_shape(
�
save_37/Assign_37Assignv/dense_2/kernelsave_37/RestoreV2:37*
T0*#
_class
loc:@v/dense_2/kernel*
use_locking(*
validate_shape(*
_output_shapes

:@
�
save_37/Assign_38Assignv/dense_2/kernel/Adamsave_37/RestoreV2:38*
_output_shapes

:@*
use_locking(*
T0*#
_class
loc:@v/dense_2/kernel*
validate_shape(
�
save_37/Assign_39Assignv/dense_2/kernel/Adam_1save_37/RestoreV2:39*
use_locking(*
_output_shapes

:@*
T0*#
_class
loc:@v/dense_2/kernel*
validate_shape(
�
save_37/restore_shardNoOp^save_37/Assign^save_37/Assign_1^save_37/Assign_10^save_37/Assign_11^save_37/Assign_12^save_37/Assign_13^save_37/Assign_14^save_37/Assign_15^save_37/Assign_16^save_37/Assign_17^save_37/Assign_18^save_37/Assign_19^save_37/Assign_2^save_37/Assign_20^save_37/Assign_21^save_37/Assign_22^save_37/Assign_23^save_37/Assign_24^save_37/Assign_25^save_37/Assign_26^save_37/Assign_27^save_37/Assign_28^save_37/Assign_29^save_37/Assign_3^save_37/Assign_30^save_37/Assign_31^save_37/Assign_32^save_37/Assign_33^save_37/Assign_34^save_37/Assign_35^save_37/Assign_36^save_37/Assign_37^save_37/Assign_38^save_37/Assign_39^save_37/Assign_4^save_37/Assign_5^save_37/Assign_6^save_37/Assign_7^save_37/Assign_8^save_37/Assign_9
3
save_37/restore_allNoOp^save_37/restore_shard
\
save_38/filename/inputConst*
_output_shapes
: *
valueB Bmodel*
dtype0
t
save_38/filenamePlaceholderWithDefaultsave_38/filename/input*
dtype0*
_output_shapes
: *
shape: 
k
save_38/ConstPlaceholderWithDefaultsave_38/filename*
shape: *
_output_shapes
: *
dtype0
�
save_38/StringJoin/inputs_1Const*
dtype0*
_output_shapes
: *<
value3B1 B+_temp_4079fb60681f4fc1a7b3541ecd3de3eb/part
~
save_38/StringJoin
StringJoinsave_38/Constsave_38/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
T
save_38/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
_
save_38/ShardedFilename/shardConst*
dtype0*
value	B : *
_output_shapes
: 
�
save_38/ShardedFilenameShardedFilenamesave_38/StringJoinsave_38/ShardedFilename/shardsave_38/num_shards*
_output_shapes
: 
�
save_38/SaveV2/tensor_namesConst*�
value�B�(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
_output_shapes
:(*
dtype0
�
save_38/SaveV2/shape_and_slicesConst*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:(
�
save_38/SaveV2SaveV2save_38/ShardedFilenamesave_38/SaveV2/tensor_namessave_38/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1v/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1v/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1v/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1v/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1v/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1v/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1*6
dtypes,
*2(
�
save_38/control_dependencyIdentitysave_38/ShardedFilename^save_38/SaveV2*
T0**
_class 
loc:@save_38/ShardedFilename*
_output_shapes
: 
�
.save_38/MergeV2Checkpoints/checkpoint_prefixesPacksave_38/ShardedFilename^save_38/control_dependency*
_output_shapes
:*
N*

axis *
T0
�
save_38/MergeV2CheckpointsMergeV2Checkpoints.save_38/MergeV2Checkpoints/checkpoint_prefixessave_38/Const*
delete_old_dirs(
�
save_38/IdentityIdentitysave_38/Const^save_38/MergeV2Checkpoints^save_38/control_dependency*
_output_shapes
: *
T0
�
save_38/RestoreV2/tensor_namesConst*
_output_shapes
:(*
dtype0*�
value�B�(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1
�
"save_38/RestoreV2/shape_and_slicesConst*
dtype0*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:(
�
save_38/RestoreV2	RestoreV2save_38/Constsave_38/RestoreV2/tensor_names"save_38/RestoreV2/shape_and_slices*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::*6
dtypes,
*2(
�
save_38/AssignAssignbeta1_powersave_38/RestoreV2*
_output_shapes
: * 
_class
loc:@pi/dense/bias*
use_locking(*
validate_shape(*
T0
�
save_38/Assign_1Assignbeta1_power_1save_38/RestoreV2:1*
use_locking(*
_output_shapes
: *
_class
loc:@v/dense/bias*
validate_shape(*
T0
�
save_38/Assign_2Assignbeta2_powersave_38/RestoreV2:2*
validate_shape(*
T0*
use_locking(* 
_class
loc:@pi/dense/bias*
_output_shapes
: 
�
save_38/Assign_3Assignbeta2_power_1save_38/RestoreV2:3*
validate_shape(*
T0*
_output_shapes
: *
use_locking(*
_class
loc:@v/dense/bias
�
save_38/Assign_4Assignpi/dense/biassave_38/RestoreV2:4*
_output_shapes
:@*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
use_locking(
�
save_38/Assign_5Assignpi/dense/bias/Adamsave_38/RestoreV2:5* 
_class
loc:@pi/dense/bias*
use_locking(*
_output_shapes
:@*
T0*
validate_shape(
�
save_38/Assign_6Assignpi/dense/bias/Adam_1save_38/RestoreV2:6*
use_locking(* 
_class
loc:@pi/dense/bias*
T0*
_output_shapes
:@*
validate_shape(
�
save_38/Assign_7Assignpi/dense/kernelsave_38/RestoreV2:7*
validate_shape(*
_output_shapes

:@*
T0*
use_locking(*"
_class
loc:@pi/dense/kernel
�
save_38/Assign_8Assignpi/dense/kernel/Adamsave_38/RestoreV2:8*
T0*
use_locking(*"
_class
loc:@pi/dense/kernel*
_output_shapes

:@*
validate_shape(
�
save_38/Assign_9Assignpi/dense/kernel/Adam_1save_38/RestoreV2:9*
_output_shapes

:@*"
_class
loc:@pi/dense/kernel*
validate_shape(*
use_locking(*
T0
�
save_38/Assign_10Assignpi/dense_1/biassave_38/RestoreV2:10*
_output_shapes
:@*
use_locking(*
validate_shape(*"
_class
loc:@pi/dense_1/bias*
T0
�
save_38/Assign_11Assignpi/dense_1/bias/Adamsave_38/RestoreV2:11*
validate_shape(*"
_class
loc:@pi/dense_1/bias*
_output_shapes
:@*
T0*
use_locking(
�
save_38/Assign_12Assignpi/dense_1/bias/Adam_1save_38/RestoreV2:12*"
_class
loc:@pi/dense_1/bias*
_output_shapes
:@*
validate_shape(*
T0*
use_locking(
�
save_38/Assign_13Assignpi/dense_1/kernelsave_38/RestoreV2:13*
_output_shapes

:@@*
validate_shape(*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel
�
save_38/Assign_14Assignpi/dense_1/kernel/Adamsave_38/RestoreV2:14*$
_class
loc:@pi/dense_1/kernel*
use_locking(*
T0*
_output_shapes

:@@*
validate_shape(
�
save_38/Assign_15Assignpi/dense_1/kernel/Adam_1save_38/RestoreV2:15*
use_locking(*
validate_shape(*$
_class
loc:@pi/dense_1/kernel*
_output_shapes

:@@*
T0
�
save_38/Assign_16Assignpi/dense_2/biassave_38/RestoreV2:16*
_output_shapes
:*
use_locking(*
T0*
validate_shape(*"
_class
loc:@pi/dense_2/bias
�
save_38/Assign_17Assignpi/dense_2/bias/Adamsave_38/RestoreV2:17*"
_class
loc:@pi/dense_2/bias*
use_locking(*
validate_shape(*
_output_shapes
:*
T0
�
save_38/Assign_18Assignpi/dense_2/bias/Adam_1save_38/RestoreV2:18*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
_output_shapes
:*
T0*
use_locking(
�
save_38/Assign_19Assignpi/dense_2/kernelsave_38/RestoreV2:19*
T0*
_output_shapes

:@*
validate_shape(*$
_class
loc:@pi/dense_2/kernel*
use_locking(
�
save_38/Assign_20Assignpi/dense_2/kernel/Adamsave_38/RestoreV2:20*$
_class
loc:@pi/dense_2/kernel*
T0*
_output_shapes

:@*
use_locking(*
validate_shape(
�
save_38/Assign_21Assignpi/dense_2/kernel/Adam_1save_38/RestoreV2:21*
use_locking(*
validate_shape(*
T0*
_output_shapes

:@*$
_class
loc:@pi/dense_2/kernel
�
save_38/Assign_22Assignv/dense/biassave_38/RestoreV2:22*
_class
loc:@v/dense/bias*
use_locking(*
_output_shapes
:@*
T0*
validate_shape(
�
save_38/Assign_23Assignv/dense/bias/Adamsave_38/RestoreV2:23*
T0*
validate_shape(*
_output_shapes
:@*
use_locking(*
_class
loc:@v/dense/bias
�
save_38/Assign_24Assignv/dense/bias/Adam_1save_38/RestoreV2:24*
validate_shape(*
use_locking(*
_output_shapes
:@*
_class
loc:@v/dense/bias*
T0
�
save_38/Assign_25Assignv/dense/kernelsave_38/RestoreV2:25*
use_locking(*
T0*
validate_shape(*!
_class
loc:@v/dense/kernel*
_output_shapes

:@
�
save_38/Assign_26Assignv/dense/kernel/Adamsave_38/RestoreV2:26*
use_locking(*!
_class
loc:@v/dense/kernel*
validate_shape(*
T0*
_output_shapes

:@
�
save_38/Assign_27Assignv/dense/kernel/Adam_1save_38/RestoreV2:27*
validate_shape(*
use_locking(*
_output_shapes

:@*!
_class
loc:@v/dense/kernel*
T0
�
save_38/Assign_28Assignv/dense_1/biassave_38/RestoreV2:28*
use_locking(*
_output_shapes
:@*!
_class
loc:@v/dense_1/bias*
validate_shape(*
T0
�
save_38/Assign_29Assignv/dense_1/bias/Adamsave_38/RestoreV2:29*
T0*!
_class
loc:@v/dense_1/bias*
validate_shape(*
_output_shapes
:@*
use_locking(
�
save_38/Assign_30Assignv/dense_1/bias/Adam_1save_38/RestoreV2:30*
_output_shapes
:@*!
_class
loc:@v/dense_1/bias*
validate_shape(*
T0*
use_locking(
�
save_38/Assign_31Assignv/dense_1/kernelsave_38/RestoreV2:31*
validate_shape(*
_output_shapes

:@@*
use_locking(*#
_class
loc:@v/dense_1/kernel*
T0
�
save_38/Assign_32Assignv/dense_1/kernel/Adamsave_38/RestoreV2:32*
validate_shape(*
_output_shapes

:@@*
use_locking(*
T0*#
_class
loc:@v/dense_1/kernel
�
save_38/Assign_33Assignv/dense_1/kernel/Adam_1save_38/RestoreV2:33*
T0*
validate_shape(*
_output_shapes

:@@*
use_locking(*#
_class
loc:@v/dense_1/kernel
�
save_38/Assign_34Assignv/dense_2/biassave_38/RestoreV2:34*!
_class
loc:@v/dense_2/bias*
T0*
_output_shapes
:*
use_locking(*
validate_shape(
�
save_38/Assign_35Assignv/dense_2/bias/Adamsave_38/RestoreV2:35*
validate_shape(*
use_locking(*
_output_shapes
:*!
_class
loc:@v/dense_2/bias*
T0
�
save_38/Assign_36Assignv/dense_2/bias/Adam_1save_38/RestoreV2:36*!
_class
loc:@v/dense_2/bias*
validate_shape(*
use_locking(*
T0*
_output_shapes
:
�
save_38/Assign_37Assignv/dense_2/kernelsave_38/RestoreV2:37*
use_locking(*
_output_shapes

:@*
T0*
validate_shape(*#
_class
loc:@v/dense_2/kernel
�
save_38/Assign_38Assignv/dense_2/kernel/Adamsave_38/RestoreV2:38*
use_locking(*#
_class
loc:@v/dense_2/kernel*
T0*
validate_shape(*
_output_shapes

:@
�
save_38/Assign_39Assignv/dense_2/kernel/Adam_1save_38/RestoreV2:39*
validate_shape(*
use_locking(*
T0*#
_class
loc:@v/dense_2/kernel*
_output_shapes

:@
�
save_38/restore_shardNoOp^save_38/Assign^save_38/Assign_1^save_38/Assign_10^save_38/Assign_11^save_38/Assign_12^save_38/Assign_13^save_38/Assign_14^save_38/Assign_15^save_38/Assign_16^save_38/Assign_17^save_38/Assign_18^save_38/Assign_19^save_38/Assign_2^save_38/Assign_20^save_38/Assign_21^save_38/Assign_22^save_38/Assign_23^save_38/Assign_24^save_38/Assign_25^save_38/Assign_26^save_38/Assign_27^save_38/Assign_28^save_38/Assign_29^save_38/Assign_3^save_38/Assign_30^save_38/Assign_31^save_38/Assign_32^save_38/Assign_33^save_38/Assign_34^save_38/Assign_35^save_38/Assign_36^save_38/Assign_37^save_38/Assign_38^save_38/Assign_39^save_38/Assign_4^save_38/Assign_5^save_38/Assign_6^save_38/Assign_7^save_38/Assign_8^save_38/Assign_9
3
save_38/restore_allNoOp^save_38/restore_shard "&E
save_38/Const:0save_38/Identity:0save_38/restore_all (5 @F8"�

trainable_variables�
�

s
pi/dense/kernel:0pi/dense/kernel/Assignpi/dense/kernel/read:02,pi/dense/kernel/Initializer/random_uniform:08
b
pi/dense/bias:0pi/dense/bias/Assignpi/dense/bias/read:02!pi/dense/bias/Initializer/zeros:08
{
pi/dense_1/kernel:0pi/dense_1/kernel/Assignpi/dense_1/kernel/read:02.pi/dense_1/kernel/Initializer/random_uniform:08
j
pi/dense_1/bias:0pi/dense_1/bias/Assignpi/dense_1/bias/read:02#pi/dense_1/bias/Initializer/zeros:08
{
pi/dense_2/kernel:0pi/dense_2/kernel/Assignpi/dense_2/kernel/read:02.pi/dense_2/kernel/Initializer/random_uniform:08
j
pi/dense_2/bias:0pi/dense_2/bias/Assignpi/dense_2/bias/read:02#pi/dense_2/bias/Initializer/zeros:08
o
v/dense/kernel:0v/dense/kernel/Assignv/dense/kernel/read:02+v/dense/kernel/Initializer/random_uniform:08
^
v/dense/bias:0v/dense/bias/Assignv/dense/bias/read:02 v/dense/bias/Initializer/zeros:08
w
v/dense_1/kernel:0v/dense_1/kernel/Assignv/dense_1/kernel/read:02-v/dense_1/kernel/Initializer/random_uniform:08
f
v/dense_1/bias:0v/dense_1/bias/Assignv/dense_1/bias/read:02"v/dense_1/bias/Initializer/zeros:08
w
v/dense_2/kernel:0v/dense_2/kernel/Assignv/dense_2/kernel/read:02-v/dense_2/kernel/Initializer/random_uniform:08
f
v/dense_2/bias:0v/dense_2/bias/Assignv/dense_2/bias/read:02"v/dense_2/bias/Initializer/zeros:08"�%
	variables�%�%
s
pi/dense/kernel:0pi/dense/kernel/Assignpi/dense/kernel/read:02,pi/dense/kernel/Initializer/random_uniform:08
b
pi/dense/bias:0pi/dense/bias/Assignpi/dense/bias/read:02!pi/dense/bias/Initializer/zeros:08
{
pi/dense_1/kernel:0pi/dense_1/kernel/Assignpi/dense_1/kernel/read:02.pi/dense_1/kernel/Initializer/random_uniform:08
j
pi/dense_1/bias:0pi/dense_1/bias/Assignpi/dense_1/bias/read:02#pi/dense_1/bias/Initializer/zeros:08
{
pi/dense_2/kernel:0pi/dense_2/kernel/Assignpi/dense_2/kernel/read:02.pi/dense_2/kernel/Initializer/random_uniform:08
j
pi/dense_2/bias:0pi/dense_2/bias/Assignpi/dense_2/bias/read:02#pi/dense_2/bias/Initializer/zeros:08
o
v/dense/kernel:0v/dense/kernel/Assignv/dense/kernel/read:02+v/dense/kernel/Initializer/random_uniform:08
^
v/dense/bias:0v/dense/bias/Assignv/dense/bias/read:02 v/dense/bias/Initializer/zeros:08
w
v/dense_1/kernel:0v/dense_1/kernel/Assignv/dense_1/kernel/read:02-v/dense_1/kernel/Initializer/random_uniform:08
f
v/dense_1/bias:0v/dense_1/bias/Assignv/dense_1/bias/read:02"v/dense_1/bias/Initializer/zeros:08
w
v/dense_2/kernel:0v/dense_2/kernel/Assignv/dense_2/kernel/read:02-v/dense_2/kernel/Initializer/random_uniform:08
f
v/dense_2/bias:0v/dense_2/bias/Assignv/dense_2/bias/read:02"v/dense_2/bias/Initializer/zeros:08
T
beta1_power:0beta1_power/Assignbeta1_power/read:02beta1_power/initial_value:0
T
beta2_power:0beta2_power/Assignbeta2_power/read:02beta2_power/initial_value:0
|
pi/dense/kernel/Adam:0pi/dense/kernel/Adam/Assignpi/dense/kernel/Adam/read:02(pi/dense/kernel/Adam/Initializer/zeros:0
�
pi/dense/kernel/Adam_1:0pi/dense/kernel/Adam_1/Assignpi/dense/kernel/Adam_1/read:02*pi/dense/kernel/Adam_1/Initializer/zeros:0
t
pi/dense/bias/Adam:0pi/dense/bias/Adam/Assignpi/dense/bias/Adam/read:02&pi/dense/bias/Adam/Initializer/zeros:0
|
pi/dense/bias/Adam_1:0pi/dense/bias/Adam_1/Assignpi/dense/bias/Adam_1/read:02(pi/dense/bias/Adam_1/Initializer/zeros:0
�
pi/dense_1/kernel/Adam:0pi/dense_1/kernel/Adam/Assignpi/dense_1/kernel/Adam/read:02*pi/dense_1/kernel/Adam/Initializer/zeros:0
�
pi/dense_1/kernel/Adam_1:0pi/dense_1/kernel/Adam_1/Assignpi/dense_1/kernel/Adam_1/read:02,pi/dense_1/kernel/Adam_1/Initializer/zeros:0
|
pi/dense_1/bias/Adam:0pi/dense_1/bias/Adam/Assignpi/dense_1/bias/Adam/read:02(pi/dense_1/bias/Adam/Initializer/zeros:0
�
pi/dense_1/bias/Adam_1:0pi/dense_1/bias/Adam_1/Assignpi/dense_1/bias/Adam_1/read:02*pi/dense_1/bias/Adam_1/Initializer/zeros:0
�
pi/dense_2/kernel/Adam:0pi/dense_2/kernel/Adam/Assignpi/dense_2/kernel/Adam/read:02*pi/dense_2/kernel/Adam/Initializer/zeros:0
�
pi/dense_2/kernel/Adam_1:0pi/dense_2/kernel/Adam_1/Assignpi/dense_2/kernel/Adam_1/read:02,pi/dense_2/kernel/Adam_1/Initializer/zeros:0
|
pi/dense_2/bias/Adam:0pi/dense_2/bias/Adam/Assignpi/dense_2/bias/Adam/read:02(pi/dense_2/bias/Adam/Initializer/zeros:0
�
pi/dense_2/bias/Adam_1:0pi/dense_2/bias/Adam_1/Assignpi/dense_2/bias/Adam_1/read:02*pi/dense_2/bias/Adam_1/Initializer/zeros:0
\
beta1_power_1:0beta1_power_1/Assignbeta1_power_1/read:02beta1_power_1/initial_value:0
\
beta2_power_1:0beta2_power_1/Assignbeta2_power_1/read:02beta2_power_1/initial_value:0
x
v/dense/kernel/Adam:0v/dense/kernel/Adam/Assignv/dense/kernel/Adam/read:02'v/dense/kernel/Adam/Initializer/zeros:0
�
v/dense/kernel/Adam_1:0v/dense/kernel/Adam_1/Assignv/dense/kernel/Adam_1/read:02)v/dense/kernel/Adam_1/Initializer/zeros:0
p
v/dense/bias/Adam:0v/dense/bias/Adam/Assignv/dense/bias/Adam/read:02%v/dense/bias/Adam/Initializer/zeros:0
x
v/dense/bias/Adam_1:0v/dense/bias/Adam_1/Assignv/dense/bias/Adam_1/read:02'v/dense/bias/Adam_1/Initializer/zeros:0
�
v/dense_1/kernel/Adam:0v/dense_1/kernel/Adam/Assignv/dense_1/kernel/Adam/read:02)v/dense_1/kernel/Adam/Initializer/zeros:0
�
v/dense_1/kernel/Adam_1:0v/dense_1/kernel/Adam_1/Assignv/dense_1/kernel/Adam_1/read:02+v/dense_1/kernel/Adam_1/Initializer/zeros:0
x
v/dense_1/bias/Adam:0v/dense_1/bias/Adam/Assignv/dense_1/bias/Adam/read:02'v/dense_1/bias/Adam/Initializer/zeros:0
�
v/dense_1/bias/Adam_1:0v/dense_1/bias/Adam_1/Assignv/dense_1/bias/Adam_1/read:02)v/dense_1/bias/Adam_1/Initializer/zeros:0
�
v/dense_2/kernel/Adam:0v/dense_2/kernel/Adam/Assignv/dense_2/kernel/Adam/read:02)v/dense_2/kernel/Adam/Initializer/zeros:0
�
v/dense_2/kernel/Adam_1:0v/dense_2/kernel/Adam_1/Assignv/dense_2/kernel/Adam_1/read:02+v/dense_2/kernel/Adam_1/Initializer/zeros:0
x
v/dense_2/bias/Adam:0v/dense_2/bias/Adam/Assignv/dense_2/bias/Adam/read:02'v/dense_2/bias/Adam/Initializer/zeros:0
�
v/dense_2/bias/Adam_1:0v/dense_2/bias/Adam_1/Assignv/dense_2/bias/Adam_1/read:02)v/dense_2/bias/Adam_1/Initializer/zeros:0"
train_op

Adam
Adam_1*�
serving_default�
)
x$
Placeholder:0���������#
v
v/Squeeze:0���������%
pi
pi/Squeeze:0	���������tensorflow/serving/predict