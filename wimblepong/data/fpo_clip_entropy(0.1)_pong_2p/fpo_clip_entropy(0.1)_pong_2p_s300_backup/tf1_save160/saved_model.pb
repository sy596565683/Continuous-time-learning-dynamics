Љ÷
Р'х&
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
2	АР
о
	ApplyAdam
var"TА	
m"TА	
v"TА
beta1_power"T
beta2_power"T
lr"T

beta1"T

beta2"T
epsilon"T	
grad"T
out"TА" 
Ttype:
2	"
use_lockingbool( "
use_nesterovbool( 
x
Assign
ref"TА

value"T

output_ref"TА"	
Ttype"
validate_shapebool("
use_lockingbool(Ш
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
Р
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
Н
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
delete_old_dirsbool(И
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
2	Р
®
Multinomial
logits"T
num_samples
output"output_dtype"
seedint "
seed2int "
Ttype:
2	" 
output_dtypetype0	:
2	И
.
Neg
x"T
y"T"
Ttype:

2	

NoOp
М
OneHot
indices"TI	
depth
on_value"T
	off_value"T
output"T"
axisint€€€€€€€€€"	
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
Н
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
list(type)(И
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	И
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
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
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
Л
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
М
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
ref"dtypeА"
shapeshape"
dtypetype"
	containerstring "
shared_namestring И
&
	ZerosLike
x"T
y"T"	
Ttype"serve*1.14.02unknown“ь
n
PlaceholderPlaceholder*
dtype0*
shape:€€€€€€€€€*'
_output_shapes
:€€€€€€€€€
h
Placeholder_1Placeholder*
dtype0*#
_output_shapes
:€€€€€€€€€*
shape:€€€€€€€€€
h
Placeholder_2Placeholder*#
_output_shapes
:€€€€€€€€€*
shape:€€€€€€€€€*
dtype0
h
Placeholder_3Placeholder*
dtype0*
shape:€€€€€€€€€*#
_output_shapes
:€€€€€€€€€
h
Placeholder_4Placeholder*
shape:€€€€€€€€€*#
_output_shapes
:€€€€€€€€€*
dtype0
•
0pi/dense/kernel/Initializer/random_uniform/shapeConst*
dtype0*
valueB"   @   *"
_class
loc:@pi/dense/kernel*
_output_shapes
:
Ч
.pi/dense/kernel/Initializer/random_uniform/minConst*
_output_shapes
: *"
_class
loc:@pi/dense/kernel*
valueB
 *феХЊ*
dtype0
Ч
.pi/dense/kernel/Initializer/random_uniform/maxConst*
valueB
 *феХ>*
_output_shapes
: *
dtype0*"
_class
loc:@pi/dense/kernel
п
8pi/dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform0pi/dense/kernel/Initializer/random_uniform/shape*
seedђ*"
_class
loc:@pi/dense/kernel*
T0*
dtype0*
seed2*
_output_shapes

:@
Џ
.pi/dense/kernel/Initializer/random_uniform/subSub.pi/dense/kernel/Initializer/random_uniform/max.pi/dense/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*"
_class
loc:@pi/dense/kernel
м
.pi/dense/kernel/Initializer/random_uniform/mulMul8pi/dense/kernel/Initializer/random_uniform/RandomUniform.pi/dense/kernel/Initializer/random_uniform/sub*"
_class
loc:@pi/dense/kernel*
_output_shapes

:@*
T0
ё
*pi/dense/kernel/Initializer/random_uniformAdd.pi/dense/kernel/Initializer/random_uniform/mul.pi/dense/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@pi/dense/kernel*
_output_shapes

:@
І
pi/dense/kernel
VariableV2*
shared_name *
shape
:@*
	container *"
_class
loc:@pi/dense/kernel*
_output_shapes

:@*
dtype0
”
pi/dense/kernel/AssignAssignpi/dense/kernel*pi/dense/kernel/Initializer/random_uniform*"
_class
loc:@pi/dense/kernel*
T0*
_output_shapes

:@*
validate_shape(*
use_locking(
~
pi/dense/kernel/readIdentitypi/dense/kernel*
T0*"
_class
loc:@pi/dense/kernel*
_output_shapes

:@
О
pi/dense/bias/Initializer/zerosConst* 
_class
loc:@pi/dense/bias*
valueB@*    *
dtype0*
_output_shapes
:@
Ы
pi/dense/bias
VariableV2*
	container *
shared_name *
_output_shapes
:@* 
_class
loc:@pi/dense/bias*
shape:@*
dtype0
Њ
pi/dense/bias/AssignAssignpi/dense/biaspi/dense/bias/Initializer/zeros* 
_class
loc:@pi/dense/bias*
_output_shapes
:@*
validate_shape(*
use_locking(*
T0
t
pi/dense/bias/readIdentitypi/dense/bias*
T0* 
_class
loc:@pi/dense/bias*
_output_shapes
:@
Ф
pi/dense/MatMulMatMulPlaceholderpi/dense/kernel/read*'
_output_shapes
:€€€€€€€€€@*
transpose_a( *
transpose_b( *
T0
Й
pi/dense/BiasAddBiasAddpi/dense/MatMulpi/dense/bias/read*'
_output_shapes
:€€€€€€€€€@*
data_formatNHWC*
T0
Y
pi/dense/TanhTanhpi/dense/BiasAdd*'
_output_shapes
:€€€€€€€€€@*
T0
©
2pi/dense_1/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*
dtype0*
valueB"@   @   *$
_class
loc:@pi/dense_1/kernel
Ы
0pi/dense_1/kernel/Initializer/random_uniform/minConst*
_output_shapes
: *
dtype0*$
_class
loc:@pi/dense_1/kernel*
valueB
 *„≥]Њ
Ы
0pi/dense_1/kernel/Initializer/random_uniform/maxConst*$
_class
loc:@pi/dense_1/kernel*
dtype0*
_output_shapes
: *
valueB
 *„≥]>
х
:pi/dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform2pi/dense_1/kernel/Initializer/random_uniform/shape*
dtype0*
seed2*
T0*$
_class
loc:@pi/dense_1/kernel*
_output_shapes

:@@*
seedђ
в
0pi/dense_1/kernel/Initializer/random_uniform/subSub0pi/dense_1/kernel/Initializer/random_uniform/max0pi/dense_1/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*$
_class
loc:@pi/dense_1/kernel
ф
0pi/dense_1/kernel/Initializer/random_uniform/mulMul:pi/dense_1/kernel/Initializer/random_uniform/RandomUniform0pi/dense_1/kernel/Initializer/random_uniform/sub*
T0*$
_class
loc:@pi/dense_1/kernel*
_output_shapes

:@@
ж
,pi/dense_1/kernel/Initializer/random_uniformAdd0pi/dense_1/kernel/Initializer/random_uniform/mul0pi/dense_1/kernel/Initializer/random_uniform/min*$
_class
loc:@pi/dense_1/kernel*
T0*
_output_shapes

:@@
Ђ
pi/dense_1/kernel
VariableV2*
shape
:@@*
	container *$
_class
loc:@pi/dense_1/kernel*
shared_name *
dtype0*
_output_shapes

:@@
џ
pi/dense_1/kernel/AssignAssignpi/dense_1/kernel,pi/dense_1/kernel/Initializer/random_uniform*
T0*
use_locking(*
validate_shape(*
_output_shapes

:@@*$
_class
loc:@pi/dense_1/kernel
Д
pi/dense_1/kernel/readIdentitypi/dense_1/kernel*$
_class
loc:@pi/dense_1/kernel*
T0*
_output_shapes

:@@
Т
!pi/dense_1/bias/Initializer/zerosConst*
dtype0*
_output_shapes
:@*"
_class
loc:@pi/dense_1/bias*
valueB@*    
Я
pi/dense_1/bias
VariableV2*
dtype0*"
_class
loc:@pi/dense_1/bias*
shared_name *
shape:@*
	container *
_output_shapes
:@
∆
pi/dense_1/bias/AssignAssignpi/dense_1/bias!pi/dense_1/bias/Initializer/zeros*
_output_shapes
:@*
T0*
validate_shape(*"
_class
loc:@pi/dense_1/bias*
use_locking(
z
pi/dense_1/bias/readIdentitypi/dense_1/bias*"
_class
loc:@pi/dense_1/bias*
T0*
_output_shapes
:@
Ъ
pi/dense_1/MatMulMatMulpi/dense/Tanhpi/dense_1/kernel/read*
transpose_b( *
T0*
transpose_a( *'
_output_shapes
:€€€€€€€€€@
П
pi/dense_1/BiasAddBiasAddpi/dense_1/MatMulpi/dense_1/bias/read*
data_formatNHWC*'
_output_shapes
:€€€€€€€€€@*
T0
]
pi/dense_1/TanhTanhpi/dense_1/BiasAdd*'
_output_shapes
:€€€€€€€€€@*
T0
©
2pi/dense_2/kernel/Initializer/random_uniform/shapeConst*
valueB"@      *
dtype0*$
_class
loc:@pi/dense_2/kernel*
_output_shapes
:
Ы
0pi/dense_2/kernel/Initializer/random_uniform/minConst*
dtype0*$
_class
loc:@pi/dense_2/kernel*
_output_shapes
: *
valueB
 *™7ЩЊ
Ы
0pi/dense_2/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *$
_class
loc:@pi/dense_2/kernel*
valueB
 *™7Щ>
х
:pi/dense_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform2pi/dense_2/kernel/Initializer/random_uniform/shape*
seedђ*$
_class
loc:@pi/dense_2/kernel*
seed2**
dtype0*
T0*
_output_shapes

:@
в
0pi/dense_2/kernel/Initializer/random_uniform/subSub0pi/dense_2/kernel/Initializer/random_uniform/max0pi/dense_2/kernel/Initializer/random_uniform/min*$
_class
loc:@pi/dense_2/kernel*
_output_shapes
: *
T0
ф
0pi/dense_2/kernel/Initializer/random_uniform/mulMul:pi/dense_2/kernel/Initializer/random_uniform/RandomUniform0pi/dense_2/kernel/Initializer/random_uniform/sub*
T0*
_output_shapes

:@*$
_class
loc:@pi/dense_2/kernel
ж
,pi/dense_2/kernel/Initializer/random_uniformAdd0pi/dense_2/kernel/Initializer/random_uniform/mul0pi/dense_2/kernel/Initializer/random_uniform/min*$
_class
loc:@pi/dense_2/kernel*
_output_shapes

:@*
T0
Ђ
pi/dense_2/kernel
VariableV2*$
_class
loc:@pi/dense_2/kernel*
shared_name *
_output_shapes

:@*
shape
:@*
	container *
dtype0
џ
pi/dense_2/kernel/AssignAssignpi/dense_2/kernel,pi/dense_2/kernel/Initializer/random_uniform*
validate_shape(*
use_locking(*
T0*
_output_shapes

:@*$
_class
loc:@pi/dense_2/kernel
Д
pi/dense_2/kernel/readIdentitypi/dense_2/kernel*
_output_shapes

:@*
T0*$
_class
loc:@pi/dense_2/kernel
Т
!pi/dense_2/bias/Initializer/zerosConst*
dtype0*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:*
valueB*    
Я
pi/dense_2/bias
VariableV2*
dtype0*"
_class
loc:@pi/dense_2/bias*
	container *
_output_shapes
:*
shared_name *
shape:
∆
pi/dense_2/bias/AssignAssignpi/dense_2/bias!pi/dense_2/bias/Initializer/zeros*
use_locking(*"
_class
loc:@pi/dense_2/bias*
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
Ь
pi/dense_2/MatMulMatMulpi/dense_1/Tanhpi/dense_2/kernel/read*
T0*
transpose_a( *
transpose_b( *'
_output_shapes
:€€€€€€€€€
П
pi/dense_2/BiasAddBiasAddpi/dense_2/MatMulpi/dense_2/bias/read*
T0*'
_output_shapes
:€€€€€€€€€*
data_formatNHWC
a
pi/LogSoftmax
LogSoftmaxpi/dense_2/BiasAdd*
T0*'
_output_shapes
:€€€€€€€€€
h
&pi/multinomial/Multinomial/num_samplesConst*
value	B :*
_output_shapes
: *
dtype0
≈
pi/multinomial/MultinomialMultinomialpi/dense_2/BiasAdd&pi/multinomial/Multinomial/num_samples*'
_output_shapes
:€€€€€€€€€*
seedђ*
seed28*
output_dtype0	*
T0
v

pi/SqueezeSqueezepi/multinomial/Multinomial*#
_output_shapes
:€€€€€€€€€*
squeeze_dims
*
T0	
X
pi/one_hot/on_valueConst*
_output_shapes
: *
valueB
 *  А?*
dtype0
Y
pi/one_hot/off_valueConst*
valueB
 *    *
dtype0*
_output_shapes
: 
R
pi/one_hot/depthConst*
value	B :*
dtype0*
_output_shapes
: 
±

pi/one_hotOneHotPlaceholder_1pi/one_hot/depthpi/one_hot/on_valuepi/one_hot/off_value*'
_output_shapes
:€€€€€€€€€*
TI0*
axis€€€€€€€€€*
T0
Z
pi/mulMul
pi/one_hotpi/LogSoftmax*
T0*'
_output_shapes
:€€€€€€€€€
Z
pi/Sum/reduction_indicesConst*
value	B :*
_output_shapes
: *
dtype0
z
pi/SumSumpi/mulpi/Sum/reduction_indices*

Tidx0*
	keep_dims( *
T0*#
_output_shapes
:€€€€€€€€€
Z
pi/one_hot_1/on_valueConst*
dtype0*
valueB
 *  А?*
_output_shapes
: 
[
pi/one_hot_1/off_valueConst*
_output_shapes
: *
valueB
 *    *
dtype0
T
pi/one_hot_1/depthConst*
_output_shapes
: *
value	B :*
dtype0
ґ
pi/one_hot_1OneHot
pi/Squeezepi/one_hot_1/depthpi/one_hot_1/on_valuepi/one_hot_1/off_value*
T0*
axis€€€€€€€€€*
TI0	*'
_output_shapes
:€€€€€€€€€
^
pi/mul_1Mulpi/one_hot_1pi/LogSoftmax*'
_output_shapes
:€€€€€€€€€*
T0
\
pi/Sum_1/reduction_indicesConst*
_output_shapes
: *
value	B :*
dtype0
А
pi/Sum_1Sumpi/mul_1pi/Sum_1/reduction_indices*

Tidx0*
	keep_dims( *#
_output_shapes
:€€€€€€€€€*
T0
£
/v/dense/kernel/Initializer/random_uniform/shapeConst*
dtype0*
valueB"   @   *!
_class
loc:@v/dense/kernel*
_output_shapes
:
Х
-v/dense/kernel/Initializer/random_uniform/minConst*
valueB
 *феХЊ*!
_class
loc:@v/dense/kernel*
dtype0*
_output_shapes
: 
Х
-v/dense/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *феХ>*!
_class
loc:@v/dense/kernel
м
7v/dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform/v/dense/kernel/Initializer/random_uniform/shape*
seed2L*
seedђ*
T0*!
_class
loc:@v/dense/kernel*
_output_shapes

:@*
dtype0
÷
-v/dense/kernel/Initializer/random_uniform/subSub-v/dense/kernel/Initializer/random_uniform/max-v/dense/kernel/Initializer/random_uniform/min*!
_class
loc:@v/dense/kernel*
T0*
_output_shapes
: 
и
-v/dense/kernel/Initializer/random_uniform/mulMul7v/dense/kernel/Initializer/random_uniform/RandomUniform-v/dense/kernel/Initializer/random_uniform/sub*
_output_shapes

:@*!
_class
loc:@v/dense/kernel*
T0
Џ
)v/dense/kernel/Initializer/random_uniformAdd-v/dense/kernel/Initializer/random_uniform/mul-v/dense/kernel/Initializer/random_uniform/min*!
_class
loc:@v/dense/kernel*
_output_shapes

:@*
T0
•
v/dense/kernel
VariableV2*
shared_name *!
_class
loc:@v/dense/kernel*
_output_shapes

:@*
shape
:@*
	container *
dtype0
ѕ
v/dense/kernel/AssignAssignv/dense/kernel)v/dense/kernel/Initializer/random_uniform*
T0*
use_locking(*!
_class
loc:@v/dense/kernel*
_output_shapes

:@*
validate_shape(
{
v/dense/kernel/readIdentityv/dense/kernel*
T0*!
_class
loc:@v/dense/kernel*
_output_shapes

:@
М
v/dense/bias/Initializer/zerosConst*
valueB@*    *
_output_shapes
:@*
dtype0*
_class
loc:@v/dense/bias
Щ
v/dense/bias
VariableV2*
shape:@*
_output_shapes
:@*
	container *
shared_name *
_class
loc:@v/dense/bias*
dtype0
Ї
v/dense/bias/AssignAssignv/dense/biasv/dense/bias/Initializer/zeros*
_class
loc:@v/dense/bias*
_output_shapes
:@*
use_locking(*
T0*
validate_shape(
q
v/dense/bias/readIdentityv/dense/bias*
_output_shapes
:@*
_class
loc:@v/dense/bias*
T0
Т
v/dense/MatMulMatMulPlaceholderv/dense/kernel/read*'
_output_shapes
:€€€€€€€€€@*
transpose_b( *
transpose_a( *
T0
Ж
v/dense/BiasAddBiasAddv/dense/MatMulv/dense/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:€€€€€€€€€@
W
v/dense/TanhTanhv/dense/BiasAdd*'
_output_shapes
:€€€€€€€€€@*
T0
І
1v/dense_1/kernel/Initializer/random_uniform/shapeConst*
valueB"@   @   *#
_class
loc:@v/dense_1/kernel*
_output_shapes
:*
dtype0
Щ
/v/dense_1/kernel/Initializer/random_uniform/minConst*#
_class
loc:@v/dense_1/kernel*
_output_shapes
: *
valueB
 *„≥]Њ*
dtype0
Щ
/v/dense_1/kernel/Initializer/random_uniform/maxConst*
dtype0*#
_class
loc:@v/dense_1/kernel*
valueB
 *„≥]>*
_output_shapes
: 
т
9v/dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform1v/dense_1/kernel/Initializer/random_uniform/shape*
seedђ*
dtype0*
T0*
_output_shapes

:@@*
seed2]*#
_class
loc:@v/dense_1/kernel
ё
/v/dense_1/kernel/Initializer/random_uniform/subSub/v/dense_1/kernel/Initializer/random_uniform/max/v/dense_1/kernel/Initializer/random_uniform/min*
T0*#
_class
loc:@v/dense_1/kernel*
_output_shapes
: 
р
/v/dense_1/kernel/Initializer/random_uniform/mulMul9v/dense_1/kernel/Initializer/random_uniform/RandomUniform/v/dense_1/kernel/Initializer/random_uniform/sub*#
_class
loc:@v/dense_1/kernel*
T0*
_output_shapes

:@@
в
+v/dense_1/kernel/Initializer/random_uniformAdd/v/dense_1/kernel/Initializer/random_uniform/mul/v/dense_1/kernel/Initializer/random_uniform/min*#
_class
loc:@v/dense_1/kernel*
_output_shapes

:@@*
T0
©
v/dense_1/kernel
VariableV2*
shape
:@@*#
_class
loc:@v/dense_1/kernel*
dtype0*
shared_name *
_output_shapes

:@@*
	container 
„
v/dense_1/kernel/AssignAssignv/dense_1/kernel+v/dense_1/kernel/Initializer/random_uniform*#
_class
loc:@v/dense_1/kernel*
validate_shape(*
T0*
_output_shapes

:@@*
use_locking(
Б
v/dense_1/kernel/readIdentityv/dense_1/kernel*#
_class
loc:@v/dense_1/kernel*
T0*
_output_shapes

:@@
Р
 v/dense_1/bias/Initializer/zerosConst*!
_class
loc:@v/dense_1/bias*
_output_shapes
:@*
valueB@*    *
dtype0
Э
v/dense_1/bias
VariableV2*
shared_name *
dtype0*
_output_shapes
:@*
shape:@*!
_class
loc:@v/dense_1/bias*
	container 
¬
v/dense_1/bias/AssignAssignv/dense_1/bias v/dense_1/bias/Initializer/zeros*
use_locking(*
validate_shape(*
T0*
_output_shapes
:@*!
_class
loc:@v/dense_1/bias
w
v/dense_1/bias/readIdentityv/dense_1/bias*!
_class
loc:@v/dense_1/bias*
T0*
_output_shapes
:@
Ч
v/dense_1/MatMulMatMulv/dense/Tanhv/dense_1/kernel/read*
transpose_a( *'
_output_shapes
:€€€€€€€€€@*
T0*
transpose_b( 
М
v/dense_1/BiasAddBiasAddv/dense_1/MatMulv/dense_1/bias/read*
data_formatNHWC*
T0*'
_output_shapes
:€€€€€€€€€@
[
v/dense_1/TanhTanhv/dense_1/BiasAdd*'
_output_shapes
:€€€€€€€€€@*
T0
І
1v/dense_2/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*
dtype0*#
_class
loc:@v/dense_2/kernel*
valueB"@      
Щ
/v/dense_2/kernel/Initializer/random_uniform/minConst*#
_class
loc:@v/dense_2/kernel*
dtype0*
valueB
 *ИОЫЊ*
_output_shapes
: 
Щ
/v/dense_2/kernel/Initializer/random_uniform/maxConst*
valueB
 *ИОЫ>*
dtype0*#
_class
loc:@v/dense_2/kernel*
_output_shapes
: 
т
9v/dense_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform1v/dense_2/kernel/Initializer/random_uniform/shape*#
_class
loc:@v/dense_2/kernel*
seed2n*
seedђ*
T0*
dtype0*
_output_shapes

:@
ё
/v/dense_2/kernel/Initializer/random_uniform/subSub/v/dense_2/kernel/Initializer/random_uniform/max/v/dense_2/kernel/Initializer/random_uniform/min*
T0*#
_class
loc:@v/dense_2/kernel*
_output_shapes
: 
р
/v/dense_2/kernel/Initializer/random_uniform/mulMul9v/dense_2/kernel/Initializer/random_uniform/RandomUniform/v/dense_2/kernel/Initializer/random_uniform/sub*
_output_shapes

:@*
T0*#
_class
loc:@v/dense_2/kernel
в
+v/dense_2/kernel/Initializer/random_uniformAdd/v/dense_2/kernel/Initializer/random_uniform/mul/v/dense_2/kernel/Initializer/random_uniform/min*
T0*
_output_shapes

:@*#
_class
loc:@v/dense_2/kernel
©
v/dense_2/kernel
VariableV2*
	container *
shared_name *#
_class
loc:@v/dense_2/kernel*
_output_shapes

:@*
shape
:@*
dtype0
„
v/dense_2/kernel/AssignAssignv/dense_2/kernel+v/dense_2/kernel/Initializer/random_uniform*
_output_shapes

:@*#
_class
loc:@v/dense_2/kernel*
T0*
validate_shape(*
use_locking(
Б
v/dense_2/kernel/readIdentityv/dense_2/kernel*#
_class
loc:@v/dense_2/kernel*
T0*
_output_shapes

:@
Р
 v/dense_2/bias/Initializer/zerosConst*!
_class
loc:@v/dense_2/bias*
valueB*    *
dtype0*
_output_shapes
:
Э
v/dense_2/bias
VariableV2*
shape:*
dtype0*
_output_shapes
:*
	container *
shared_name *!
_class
loc:@v/dense_2/bias
¬
v/dense_2/bias/AssignAssignv/dense_2/bias v/dense_2/bias/Initializer/zeros*
T0*
_output_shapes
:*
validate_shape(*
use_locking(*!
_class
loc:@v/dense_2/bias
w
v/dense_2/bias/readIdentityv/dense_2/bias*
_output_shapes
:*
T0*!
_class
loc:@v/dense_2/bias
Щ
v/dense_2/MatMulMatMulv/dense_1/Tanhv/dense_2/kernel/read*'
_output_shapes
:€€€€€€€€€*
transpose_a( *
T0*
transpose_b( 
М
v/dense_2/BiasAddBiasAddv/dense_2/MatMulv/dense_2/bias/read*
data_formatNHWC*'
_output_shapes
:€€€€€€€€€*
T0
l
	v/SqueezeSqueezev/dense_2/BiasAdd*
squeeze_dims
*#
_output_shapes
:€€€€€€€€€*
T0
O
subSubpi/SumPlaceholder_4*
T0*#
_output_shapes
:€€€€€€€€€
=
ExpExpsub*
T0*#
_output_shapes
:€€€€€€€€€
N
	Greater/yConst*
dtype0*
valueB
 *    *
_output_shapes
: 
Z
GreaterGreaterPlaceholder_2	Greater/y*#
_output_shapes
:€€€€€€€€€*
T0
J
mul/xConst*
dtype0*
valueB
 *ЪЩЩ?*
_output_shapes
: 
N
mulMulmul/xPlaceholder_2*
T0*#
_output_shapes
:€€€€€€€€€
L
mul_1/xConst*
dtype0*
valueB
 *ЌћL?*
_output_shapes
: 
R
mul_1Mulmul_1/xPlaceholder_2*#
_output_shapes
:€€€€€€€€€*
T0
S
SelectSelectGreatermulmul_1*#
_output_shapes
:€€€€€€€€€*
T0
N
mul_2MulExpPlaceholder_2*#
_output_shapes
:€€€€€€€€€*
T0
O
MinimumMinimummul_2Select*
T0*#
_output_shapes
:€€€€€€€€€
O
ConstConst*
dtype0*
valueB: *
_output_shapes
:
Z
MeanMeanMinimumConst*
T0*
	keep_dims( *

Tidx0*
_output_shapes
: 
1
NegNegMean*
_output_shapes
: *
T0
T
sub_1SubPlaceholder_3	v/Squeeze*
T0*#
_output_shapes
:€€€€€€€€€
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
:€€€€€€€€€
Q
Const_1Const*
dtype0*
_output_shapes
:*
valueB: 
Z
Mean_1MeanpowConst_1*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
Q
sub_2SubPlaceholder_4pi/Sum*#
_output_shapes
:€€€€€€€€€*
T0
Q
Const_2Const*
_output_shapes
:*
dtype0*
valueB: 
\
Mean_2Meansub_2Const_2*

Tidx0*
	keep_dims( *
_output_shapes
: *
T0
M
Neg_1Negpi/LogSoftmax*
T0*'
_output_shapes
:€€€€€€€€€
M
Exp_1Exppi/LogSoftmax*'
_output_shapes
:€€€€€€€€€*
T0
L
mul_3MulNeg_1Exp_1*
T0*'
_output_shapes
:€€€€€€€€€
X
Const_3Const*
valueB"       *
dtype0*
_output_shapes
:
\
Mean_3Meanmul_3Const_3*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
L
mul_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@
>
mul_4MulMean_3mul_4/y*
T0*
_output_shapes
: 
P
Greater_1/yConst*
valueB
 *ЪЩЩ?*
_output_shapes
: *
dtype0
T
	Greater_1GreaterExpGreater_1/y*
T0*#
_output_shapes
:€€€€€€€€€
K
Less/yConst*
valueB
 *ЌћL?*
dtype0*
_output_shapes
: 
G
LessLessExpLess/y*#
_output_shapes
:€€€€€€€€€*
T0
L
	LogicalOr	LogicalOr	Greater_1Less*#
_output_shapes
:€€€€€€€€€
d
CastCast	LogicalOr*

SrcT0
*
Truncate( *#
_output_shapes
:€€€€€€€€€*

DstT0
Q
Const_4Const*
dtype0*
_output_shapes
:*
valueB: 
[
Mean_4MeanCastConst_4*
_output_shapes
: *
	keep_dims( *
T0*

Tidx0
L
mul_5/xConst*
valueB
 *Ќћћ=*
dtype0*
_output_shapes
: 
=
mul_5Mulmul_5/xmul_4*
T0*
_output_shapes
: 
9
sub_3SubNegmul_5*
_output_shapes
: *
T0
R
gradients/ShapeConst*
_output_shapes
: *
dtype0*
valueB 
X
gradients/grad_ys_0Const*
dtype0*
valueB
 *  А?*
_output_shapes
: 
o
gradients/FillFillgradients/Shapegradients/grad_ys_0*
_output_shapes
: *
T0*

index_type0
P
gradients/sub_3_grad/NegNeggradients/Fill*
_output_shapes
: *
T0
Y
%gradients/sub_3_grad/tuple/group_depsNoOp^gradients/Fill^gradients/sub_3_grad/Neg
µ
-gradients/sub_3_grad/tuple/control_dependencyIdentitygradients/Fill&^gradients/sub_3_grad/tuple/group_deps*
_output_shapes
: *
T0*!
_class
loc:@gradients/Fill
Ћ
/gradients/sub_3_grad/tuple/control_dependency_1Identitygradients/sub_3_grad/Neg&^gradients/sub_3_grad/tuple/group_deps*
T0*
_output_shapes
: *+
_class!
loc:@gradients/sub_3_grad/Neg
m
gradients/Neg_grad/NegNeg-gradients/sub_3_grad/tuple/control_dependency*
_output_shapes
: *
T0
x
gradients/mul_5_grad/MulMul/gradients/sub_3_grad/tuple/control_dependency_1mul_4*
T0*
_output_shapes
: 
|
gradients/mul_5_grad/Mul_1Mul/gradients/sub_3_grad/tuple/control_dependency_1mul_5/x*
T0*
_output_shapes
: 
e
%gradients/mul_5_grad/tuple/group_depsNoOp^gradients/mul_5_grad/Mul^gradients/mul_5_grad/Mul_1
…
-gradients/mul_5_grad/tuple/control_dependencyIdentitygradients/mul_5_grad/Mul&^gradients/mul_5_grad/tuple/group_deps*
_output_shapes
: *
T0*+
_class!
loc:@gradients/mul_5_grad/Mul
ѕ
/gradients/mul_5_grad/tuple/control_dependency_1Identitygradients/mul_5_grad/Mul_1&^gradients/mul_5_grad/tuple/group_deps*-
_class#
!loc:@gradients/mul_5_grad/Mul_1*
T0*
_output_shapes
: 
k
!gradients/Mean_grad/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB:
Ф
gradients/Mean_grad/ReshapeReshapegradients/Neg_grad/Neg!gradients/Mean_grad/Reshape/shape*
Tshape0*
_output_shapes
:*
T0
`
gradients/Mean_grad/ShapeShapeMinimum*
T0*
_output_shapes
:*
out_type0
Ш
gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Shape*#
_output_shapes
:€€€€€€€€€*
T0*

Tmultiples0
b
gradients/Mean_grad/Shape_1ShapeMinimum*
_output_shapes
:*
T0*
out_type0
^
gradients/Mean_grad/Shape_2Const*
_output_shapes
: *
dtype0*
valueB 
c
gradients/Mean_grad/ConstConst*
valueB: *
_output_shapes
:*
dtype0
Ц
gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_1gradients/Mean_grad/Const*

Tidx0*
	keep_dims( *
_output_shapes
: *
T0
e
gradients/Mean_grad/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
Ъ
gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_2gradients/Mean_grad/Const_1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
_
gradients/Mean_grad/Maximum/yConst*
dtype0*
value	B :*
_output_shapes
: 
В
gradients/Mean_grad/MaximumMaximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum/y*
T0*
_output_shapes
: 
А
gradients/Mean_grad/floordivFloorDivgradients/Mean_grad/Prodgradients/Mean_grad/Maximum*
T0*
_output_shapes
: 
~
gradients/Mean_grad/CastCastgradients/Mean_grad/floordiv*

SrcT0*
_output_shapes
: *

DstT0*
Truncate( 
И
gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*
T0*#
_output_shapes
:€€€€€€€€€
z
gradients/mul_4_grad/MulMul/gradients/mul_5_grad/tuple/control_dependency_1mul_4/y*
T0*
_output_shapes
: 
{
gradients/mul_4_grad/Mul_1Mul/gradients/mul_5_grad/tuple/control_dependency_1Mean_3*
T0*
_output_shapes
: 
e
%gradients/mul_4_grad/tuple/group_depsNoOp^gradients/mul_4_grad/Mul^gradients/mul_4_grad/Mul_1
…
-gradients/mul_4_grad/tuple/control_dependencyIdentitygradients/mul_4_grad/Mul&^gradients/mul_4_grad/tuple/group_deps*
T0*+
_class!
loc:@gradients/mul_4_grad/Mul*
_output_shapes
: 
ѕ
/gradients/mul_4_grad/tuple/control_dependency_1Identitygradients/mul_4_grad/Mul_1&^gradients/mul_4_grad/tuple/group_deps*-
_class#
!loc:@gradients/mul_4_grad/Mul_1*
T0*
_output_shapes
: 
a
gradients/Minimum_grad/ShapeShapemul_2*
T0*
_output_shapes
:*
out_type0
d
gradients/Minimum_grad/Shape_1ShapeSelect*
_output_shapes
:*
out_type0*
T0
y
gradients/Minimum_grad/Shape_2Shapegradients/Mean_grad/truediv*
_output_shapes
:*
T0*
out_type0
g
"gradients/Minimum_grad/zeros/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0
®
gradients/Minimum_grad/zerosFillgradients/Minimum_grad/Shape_2"gradients/Minimum_grad/zeros/Const*

index_type0*
T0*#
_output_shapes
:€€€€€€€€€
j
 gradients/Minimum_grad/LessEqual	LessEqualmul_2Select*#
_output_shapes
:€€€€€€€€€*
T0
ј
,gradients/Minimum_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Minimum_grad/Shapegradients/Minimum_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
≤
gradients/Minimum_grad/SelectSelect gradients/Minimum_grad/LessEqualgradients/Mean_grad/truedivgradients/Minimum_grad/zeros*
T0*#
_output_shapes
:€€€€€€€€€
Ѓ
gradients/Minimum_grad/SumSumgradients/Minimum_grad/Select,gradients/Minimum_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
T0*
	keep_dims( 
Я
gradients/Minimum_grad/ReshapeReshapegradients/Minimum_grad/Sumgradients/Minimum_grad/Shape*
T0*
Tshape0*#
_output_shapes
:€€€€€€€€€
і
gradients/Minimum_grad/Select_1Select gradients/Minimum_grad/LessEqualgradients/Minimum_grad/zerosgradients/Mean_grad/truediv*#
_output_shapes
:€€€€€€€€€*
T0
і
gradients/Minimum_grad/Sum_1Sumgradients/Minimum_grad/Select_1.gradients/Minimum_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
•
 gradients/Minimum_grad/Reshape_1Reshapegradients/Minimum_grad/Sum_1gradients/Minimum_grad/Shape_1*
Tshape0*
T0*#
_output_shapes
:€€€€€€€€€
s
'gradients/Minimum_grad/tuple/group_depsNoOp^gradients/Minimum_grad/Reshape!^gradients/Minimum_grad/Reshape_1
ж
/gradients/Minimum_grad/tuple/control_dependencyIdentitygradients/Minimum_grad/Reshape(^gradients/Minimum_grad/tuple/group_deps*#
_output_shapes
:€€€€€€€€€*
T0*1
_class'
%#loc:@gradients/Minimum_grad/Reshape
м
1gradients/Minimum_grad/tuple/control_dependency_1Identity gradients/Minimum_grad/Reshape_1(^gradients/Minimum_grad/tuple/group_deps*3
_class)
'%loc:@gradients/Minimum_grad/Reshape_1*#
_output_shapes
:€€€€€€€€€*
T0
t
#gradients/Mean_3_grad/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      
≥
gradients/Mean_3_grad/ReshapeReshape-gradients/mul_4_grad/tuple/control_dependency#gradients/Mean_3_grad/Reshape/shape*
T0*
_output_shapes

:*
Tshape0
`
gradients/Mean_3_grad/ShapeShapemul_3*
out_type0*
T0*
_output_shapes
:
Ґ
gradients/Mean_3_grad/TileTilegradients/Mean_3_grad/Reshapegradients/Mean_3_grad/Shape*'
_output_shapes
:€€€€€€€€€*
T0*

Tmultiples0
b
gradients/Mean_3_grad/Shape_1Shapemul_3*
out_type0*
_output_shapes
:*
T0
`
gradients/Mean_3_grad/Shape_2Const*
valueB *
_output_shapes
: *
dtype0
e
gradients/Mean_3_grad/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
Ь
gradients/Mean_3_grad/ProdProdgradients/Mean_3_grad/Shape_1gradients/Mean_3_grad/Const*
	keep_dims( *
_output_shapes
: *
T0*

Tidx0
g
gradients/Mean_3_grad/Const_1Const*
dtype0*
valueB: *
_output_shapes
:
†
gradients/Mean_3_grad/Prod_1Prodgradients/Mean_3_grad/Shape_2gradients/Mean_3_grad/Const_1*

Tidx0*
	keep_dims( *
_output_shapes
: *
T0
a
gradients/Mean_3_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
И
gradients/Mean_3_grad/MaximumMaximumgradients/Mean_3_grad/Prod_1gradients/Mean_3_grad/Maximum/y*
_output_shapes
: *
T0
Ж
gradients/Mean_3_grad/floordivFloorDivgradients/Mean_3_grad/Prodgradients/Mean_3_grad/Maximum*
T0*
_output_shapes
: 
В
gradients/Mean_3_grad/CastCastgradients/Mean_3_grad/floordiv*

SrcT0*

DstT0*
Truncate( *
_output_shapes
: 
Т
gradients/Mean_3_grad/truedivRealDivgradients/Mean_3_grad/Tilegradients/Mean_3_grad/Cast*
T0*'
_output_shapes
:€€€€€€€€€
]
gradients/mul_2_grad/ShapeShapeExp*
out_type0*
T0*
_output_shapes
:
i
gradients/mul_2_grad/Shape_1ShapePlaceholder_2*
T0*
_output_shapes
:*
out_type0
Ї
*gradients/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/mul_2_grad/Shapegradients/mul_2_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0
Н
gradients/mul_2_grad/MulMul/gradients/Minimum_grad/tuple/control_dependencyPlaceholder_2*
T0*#
_output_shapes
:€€€€€€€€€
•
gradients/mul_2_grad/SumSumgradients/mul_2_grad/Mul*gradients/mul_2_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
T0*
	keep_dims( 
Щ
gradients/mul_2_grad/ReshapeReshapegradients/mul_2_grad/Sumgradients/mul_2_grad/Shape*
T0*#
_output_shapes
:€€€€€€€€€*
Tshape0
Е
gradients/mul_2_grad/Mul_1MulExp/gradients/Minimum_grad/tuple/control_dependency*
T0*#
_output_shapes
:€€€€€€€€€
Ђ
gradients/mul_2_grad/Sum_1Sumgradients/mul_2_grad/Mul_1,gradients/mul_2_grad/BroadcastGradientArgs:1*
	keep_dims( *
_output_shapes
:*
T0*

Tidx0
Я
gradients/mul_2_grad/Reshape_1Reshapegradients/mul_2_grad/Sum_1gradients/mul_2_grad/Shape_1*
Tshape0*#
_output_shapes
:€€€€€€€€€*
T0
m
%gradients/mul_2_grad/tuple/group_depsNoOp^gradients/mul_2_grad/Reshape^gradients/mul_2_grad/Reshape_1
ё
-gradients/mul_2_grad/tuple/control_dependencyIdentitygradients/mul_2_grad/Reshape&^gradients/mul_2_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/mul_2_grad/Reshape*#
_output_shapes
:€€€€€€€€€
д
/gradients/mul_2_grad/tuple/control_dependency_1Identitygradients/mul_2_grad/Reshape_1&^gradients/mul_2_grad/tuple/group_deps*#
_output_shapes
:€€€€€€€€€*1
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
gradients/mul_3_grad/Shape_1ShapeExp_1*
out_type0*
T0*
_output_shapes
:
Ї
*gradients/mul_3_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/mul_3_grad/Shapegradients/mul_3_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
w
gradients/mul_3_grad/MulMulgradients/Mean_3_grad/truedivExp_1*
T0*'
_output_shapes
:€€€€€€€€€
•
gradients/mul_3_grad/SumSumgradients/mul_3_grad/Mul*gradients/mul_3_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
Э
gradients/mul_3_grad/ReshapeReshapegradients/mul_3_grad/Sumgradients/mul_3_grad/Shape*
Tshape0*'
_output_shapes
:€€€€€€€€€*
T0
y
gradients/mul_3_grad/Mul_1MulNeg_1gradients/Mean_3_grad/truediv*'
_output_shapes
:€€€€€€€€€*
T0
Ђ
gradients/mul_3_grad/Sum_1Sumgradients/mul_3_grad/Mul_1,gradients/mul_3_grad/BroadcastGradientArgs:1*

Tidx0*
_output_shapes
:*
T0*
	keep_dims( 
£
gradients/mul_3_grad/Reshape_1Reshapegradients/mul_3_grad/Sum_1gradients/mul_3_grad/Shape_1*
Tshape0*'
_output_shapes
:€€€€€€€€€*
T0
m
%gradients/mul_3_grad/tuple/group_depsNoOp^gradients/mul_3_grad/Reshape^gradients/mul_3_grad/Reshape_1
в
-gradients/mul_3_grad/tuple/control_dependencyIdentitygradients/mul_3_grad/Reshape&^gradients/mul_3_grad/tuple/group_deps*'
_output_shapes
:€€€€€€€€€*/
_class%
#!loc:@gradients/mul_3_grad/Reshape*
T0
и
/gradients/mul_3_grad/tuple/control_dependency_1Identitygradients/mul_3_grad/Reshape_1&^gradients/mul_3_grad/tuple/group_deps*'
_output_shapes
:€€€€€€€€€*1
_class'
%#loc:@gradients/mul_3_grad/Reshape_1*
T0

gradients/Exp_grad/mulMul-gradients/mul_2_grad/tuple/control_dependencyExp*#
_output_shapes
:€€€€€€€€€*
T0
А
gradients/Neg_1_grad/NegNeg-gradients/mul_3_grad/tuple/control_dependency*'
_output_shapes
:€€€€€€€€€*
T0
Й
gradients/Exp_1_grad/mulMul/gradients/mul_3_grad/tuple/control_dependency_1Exp_1*
T0*'
_output_shapes
:€€€€€€€€€
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
і
(gradients/sub_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/sub_grad/Shapegradients/sub_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0
Я
gradients/sub_grad/SumSumgradients/Exp_grad/mul(gradients/sub_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
_output_shapes
:*
T0
У
gradients/sub_grad/ReshapeReshapegradients/sub_grad/Sumgradients/sub_grad/Shape*#
_output_shapes
:€€€€€€€€€*
Tshape0*
T0
£
gradients/sub_grad/Sum_1Sumgradients/Exp_grad/mul*gradients/sub_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Z
gradients/sub_grad/NegNeggradients/sub_grad/Sum_1*
T0*
_output_shapes
:
Ч
gradients/sub_grad/Reshape_1Reshapegradients/sub_grad/Neggradients/sub_grad/Shape_1*
T0*#
_output_shapes
:€€€€€€€€€*
Tshape0
g
#gradients/sub_grad/tuple/group_depsNoOp^gradients/sub_grad/Reshape^gradients/sub_grad/Reshape_1
÷
+gradients/sub_grad/tuple/control_dependencyIdentitygradients/sub_grad/Reshape$^gradients/sub_grad/tuple/group_deps*-
_class#
!loc:@gradients/sub_grad/Reshape*#
_output_shapes
:€€€€€€€€€*
T0
№
-gradients/sub_grad/tuple/control_dependency_1Identitygradients/sub_grad/Reshape_1$^gradients/sub_grad/tuple/group_deps*#
_output_shapes
:€€€€€€€€€*
T0*/
_class%
#!loc:@gradients/sub_grad/Reshape_1
a
gradients/pi/Sum_grad/ShapeShapepi/mul*
out_type0*
_output_shapes
:*
T0
М
gradients/pi/Sum_grad/SizeConst*
value	B :*
_output_shapes
: *.
_class$
" loc:@gradients/pi/Sum_grad/Shape*
dtype0
І
gradients/pi/Sum_grad/addAddpi/Sum/reduction_indicesgradients/pi/Sum_grad/Size*.
_class$
" loc:@gradients/pi/Sum_grad/Shape*
_output_shapes
: *
T0
≠
gradients/pi/Sum_grad/modFloorModgradients/pi/Sum_grad/addgradients/pi/Sum_grad/Size*.
_class$
" loc:@gradients/pi/Sum_grad/Shape*
_output_shapes
: *
T0
Р
gradients/pi/Sum_grad/Shape_1Const*.
_class$
" loc:@gradients/pi/Sum_grad/Shape*
valueB *
dtype0*
_output_shapes
: 
У
!gradients/pi/Sum_grad/range/startConst*
_output_shapes
: *
dtype0*.
_class$
" loc:@gradients/pi/Sum_grad/Shape*
value	B : 
У
!gradients/pi/Sum_grad/range/deltaConst*
value	B :*
_output_shapes
: *
dtype0*.
_class$
" loc:@gradients/pi/Sum_grad/Shape
ё
gradients/pi/Sum_grad/rangeRange!gradients/pi/Sum_grad/range/startgradients/pi/Sum_grad/Size!gradients/pi/Sum_grad/range/delta*

Tidx0*.
_class$
" loc:@gradients/pi/Sum_grad/Shape*
_output_shapes
:
Т
 gradients/pi/Sum_grad/Fill/valueConst*
value	B :*
dtype0*
_output_shapes
: *.
_class$
" loc:@gradients/pi/Sum_grad/Shape
∆
gradients/pi/Sum_grad/FillFillgradients/pi/Sum_grad/Shape_1 gradients/pi/Sum_grad/Fill/value*

index_type0*
T0*
_output_shapes
: *.
_class$
" loc:@gradients/pi/Sum_grad/Shape
Г
#gradients/pi/Sum_grad/DynamicStitchDynamicStitchgradients/pi/Sum_grad/rangegradients/pi/Sum_grad/modgradients/pi/Sum_grad/Shapegradients/pi/Sum_grad/Fill*
_output_shapes
:*
T0*
N*.
_class$
" loc:@gradients/pi/Sum_grad/Shape
С
gradients/pi/Sum_grad/Maximum/yConst*.
_class$
" loc:@gradients/pi/Sum_grad/Shape*
value	B :*
_output_shapes
: *
dtype0
√
gradients/pi/Sum_grad/MaximumMaximum#gradients/pi/Sum_grad/DynamicStitchgradients/pi/Sum_grad/Maximum/y*
T0*
_output_shapes
:*.
_class$
" loc:@gradients/pi/Sum_grad/Shape
ї
gradients/pi/Sum_grad/floordivFloorDivgradients/pi/Sum_grad/Shapegradients/pi/Sum_grad/Maximum*.
_class$
" loc:@gradients/pi/Sum_grad/Shape*
T0*
_output_shapes
:
√
gradients/pi/Sum_grad/ReshapeReshape+gradients/sub_grad/tuple/control_dependency#gradients/pi/Sum_grad/DynamicStitch*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€*
Tshape0
•
gradients/pi/Sum_grad/TileTilegradients/pi/Sum_grad/Reshapegradients/pi/Sum_grad/floordiv*

Tmultiples0*
T0*'
_output_shapes
:€€€€€€€€€
e
gradients/pi/mul_grad/ShapeShape
pi/one_hot*
out_type0*
T0*
_output_shapes
:
j
gradients/pi/mul_grad/Shape_1Shapepi/LogSoftmax*
out_type0*
_output_shapes
:*
T0
љ
+gradients/pi/mul_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/pi/mul_grad/Shapegradients/pi/mul_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
}
gradients/pi/mul_grad/MulMulgradients/pi/Sum_grad/Tilepi/LogSoftmax*'
_output_shapes
:€€€€€€€€€*
T0
®
gradients/pi/mul_grad/SumSumgradients/pi/mul_grad/Mul+gradients/pi/mul_grad/BroadcastGradientArgs*
	keep_dims( *
_output_shapes
:*

Tidx0*
T0
†
gradients/pi/mul_grad/ReshapeReshapegradients/pi/mul_grad/Sumgradients/pi/mul_grad/Shape*'
_output_shapes
:€€€€€€€€€*
T0*
Tshape0
|
gradients/pi/mul_grad/Mul_1Mul
pi/one_hotgradients/pi/Sum_grad/Tile*'
_output_shapes
:€€€€€€€€€*
T0
Ѓ
gradients/pi/mul_grad/Sum_1Sumgradients/pi/mul_grad/Mul_1-gradients/pi/mul_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
¶
gradients/pi/mul_grad/Reshape_1Reshapegradients/pi/mul_grad/Sum_1gradients/pi/mul_grad/Shape_1*
T0*'
_output_shapes
:€€€€€€€€€*
Tshape0
p
&gradients/pi/mul_grad/tuple/group_depsNoOp^gradients/pi/mul_grad/Reshape ^gradients/pi/mul_grad/Reshape_1
ж
.gradients/pi/mul_grad/tuple/control_dependencyIdentitygradients/pi/mul_grad/Reshape'^gradients/pi/mul_grad/tuple/group_deps*'
_output_shapes
:€€€€€€€€€*
T0*0
_class&
$"loc:@gradients/pi/mul_grad/Reshape
м
0gradients/pi/mul_grad/tuple/control_dependency_1Identitygradients/pi/mul_grad/Reshape_1'^gradients/pi/mul_grad/tuple/group_deps*'
_output_shapes
:€€€€€€€€€*
T0*2
_class(
&$loc:@gradients/pi/mul_grad/Reshape_1
д
gradients/AddNAddNgradients/Neg_1_grad/Neggradients/Exp_1_grad/mul0gradients/pi/mul_grad/tuple/control_dependency_1*+
_class!
loc:@gradients/Neg_1_grad/Neg*
T0*'
_output_shapes
:€€€€€€€€€*
N
h
 gradients/pi/LogSoftmax_grad/ExpExppi/LogSoftmax*'
_output_shapes
:€€€€€€€€€*
T0
}
2gradients/pi/LogSoftmax_grad/Sum/reduction_indicesConst*
valueB :
€€€€€€€€€*
_output_shapes
: *
dtype0
Ї
 gradients/pi/LogSoftmax_grad/SumSumgradients/AddN2gradients/pi/LogSoftmax_grad/Sum/reduction_indices*'
_output_shapes
:€€€€€€€€€*

Tidx0*
	keep_dims(*
T0
Э
 gradients/pi/LogSoftmax_grad/mulMul gradients/pi/LogSoftmax_grad/Sum gradients/pi/LogSoftmax_grad/Exp*
T0*'
_output_shapes
:€€€€€€€€€
Л
 gradients/pi/LogSoftmax_grad/subSubgradients/AddN gradients/pi/LogSoftmax_grad/mul*
T0*'
_output_shapes
:€€€€€€€€€
Ъ
-gradients/pi/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad gradients/pi/LogSoftmax_grad/sub*
T0*
_output_shapes
:*
data_formatNHWC
Н
2gradients/pi/dense_2/BiasAdd_grad/tuple/group_depsNoOp!^gradients/pi/LogSoftmax_grad/sub.^gradients/pi/dense_2/BiasAdd_grad/BiasAddGrad
Д
:gradients/pi/dense_2/BiasAdd_grad/tuple/control_dependencyIdentity gradients/pi/LogSoftmax_grad/sub3^gradients/pi/dense_2/BiasAdd_grad/tuple/group_deps*
T0*'
_output_shapes
:€€€€€€€€€*3
_class)
'%loc:@gradients/pi/LogSoftmax_grad/sub
У
<gradients/pi/dense_2/BiasAdd_grad/tuple/control_dependency_1Identity-gradients/pi/dense_2/BiasAdd_grad/BiasAddGrad3^gradients/pi/dense_2/BiasAdd_grad/tuple/group_deps*
_output_shapes
:*
T0*@
_class6
42loc:@gradients/pi/dense_2/BiasAdd_grad/BiasAddGrad
Ё
'gradients/pi/dense_2/MatMul_grad/MatMulMatMul:gradients/pi/dense_2/BiasAdd_grad/tuple/control_dependencypi/dense_2/kernel/read*'
_output_shapes
:€€€€€€€€€@*
transpose_a( *
transpose_b(*
T0
ѕ
)gradients/pi/dense_2/MatMul_grad/MatMul_1MatMulpi/dense_1/Tanh:gradients/pi/dense_2/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
_output_shapes

:@*
T0*
transpose_a(
П
1gradients/pi/dense_2/MatMul_grad/tuple/group_depsNoOp(^gradients/pi/dense_2/MatMul_grad/MatMul*^gradients/pi/dense_2/MatMul_grad/MatMul_1
Р
9gradients/pi/dense_2/MatMul_grad/tuple/control_dependencyIdentity'gradients/pi/dense_2/MatMul_grad/MatMul2^gradients/pi/dense_2/MatMul_grad/tuple/group_deps*'
_output_shapes
:€€€€€€€€€@*:
_class0
.,loc:@gradients/pi/dense_2/MatMul_grad/MatMul*
T0
Н
;gradients/pi/dense_2/MatMul_grad/tuple/control_dependency_1Identity)gradients/pi/dense_2/MatMul_grad/MatMul_12^gradients/pi/dense_2/MatMul_grad/tuple/group_deps*
T0*
_output_shapes

:@*<
_class2
0.loc:@gradients/pi/dense_2/MatMul_grad/MatMul_1
±
'gradients/pi/dense_1/Tanh_grad/TanhGradTanhGradpi/dense_1/Tanh9gradients/pi/dense_2/MatMul_grad/tuple/control_dependency*
T0*'
_output_shapes
:€€€€€€€€€@
°
-gradients/pi/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad'gradients/pi/dense_1/Tanh_grad/TanhGrad*
data_formatNHWC*
T0*
_output_shapes
:@
Ф
2gradients/pi/dense_1/BiasAdd_grad/tuple/group_depsNoOp.^gradients/pi/dense_1/BiasAdd_grad/BiasAddGrad(^gradients/pi/dense_1/Tanh_grad/TanhGrad
Т
:gradients/pi/dense_1/BiasAdd_grad/tuple/control_dependencyIdentity'gradients/pi/dense_1/Tanh_grad/TanhGrad3^gradients/pi/dense_1/BiasAdd_grad/tuple/group_deps*'
_output_shapes
:€€€€€€€€€@*:
_class0
.,loc:@gradients/pi/dense_1/Tanh_grad/TanhGrad*
T0
У
<gradients/pi/dense_1/BiasAdd_grad/tuple/control_dependency_1Identity-gradients/pi/dense_1/BiasAdd_grad/BiasAddGrad3^gradients/pi/dense_1/BiasAdd_grad/tuple/group_deps*
T0*@
_class6
42loc:@gradients/pi/dense_1/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@
Ё
'gradients/pi/dense_1/MatMul_grad/MatMulMatMul:gradients/pi/dense_1/BiasAdd_grad/tuple/control_dependencypi/dense_1/kernel/read*
transpose_b(*
transpose_a( *
T0*'
_output_shapes
:€€€€€€€€€@
Ќ
)gradients/pi/dense_1/MatMul_grad/MatMul_1MatMulpi/dense/Tanh:gradients/pi/dense_1/BiasAdd_grad/tuple/control_dependency*
_output_shapes

:@@*
T0*
transpose_a(*
transpose_b( 
П
1gradients/pi/dense_1/MatMul_grad/tuple/group_depsNoOp(^gradients/pi/dense_1/MatMul_grad/MatMul*^gradients/pi/dense_1/MatMul_grad/MatMul_1
Р
9gradients/pi/dense_1/MatMul_grad/tuple/control_dependencyIdentity'gradients/pi/dense_1/MatMul_grad/MatMul2^gradients/pi/dense_1/MatMul_grad/tuple/group_deps*'
_output_shapes
:€€€€€€€€€@*:
_class0
.,loc:@gradients/pi/dense_1/MatMul_grad/MatMul*
T0
Н
;gradients/pi/dense_1/MatMul_grad/tuple/control_dependency_1Identity)gradients/pi/dense_1/MatMul_grad/MatMul_12^gradients/pi/dense_1/MatMul_grad/tuple/group_deps*<
_class2
0.loc:@gradients/pi/dense_1/MatMul_grad/MatMul_1*
T0*
_output_shapes

:@@
≠
%gradients/pi/dense/Tanh_grad/TanhGradTanhGradpi/dense/Tanh9gradients/pi/dense_1/MatMul_grad/tuple/control_dependency*
T0*'
_output_shapes
:€€€€€€€€€@
Э
+gradients/pi/dense/BiasAdd_grad/BiasAddGradBiasAddGrad%gradients/pi/dense/Tanh_grad/TanhGrad*
data_formatNHWC*
_output_shapes
:@*
T0
О
0gradients/pi/dense/BiasAdd_grad/tuple/group_depsNoOp,^gradients/pi/dense/BiasAdd_grad/BiasAddGrad&^gradients/pi/dense/Tanh_grad/TanhGrad
К
8gradients/pi/dense/BiasAdd_grad/tuple/control_dependencyIdentity%gradients/pi/dense/Tanh_grad/TanhGrad1^gradients/pi/dense/BiasAdd_grad/tuple/group_deps*8
_class.
,*loc:@gradients/pi/dense/Tanh_grad/TanhGrad*'
_output_shapes
:€€€€€€€€€@*
T0
Л
:gradients/pi/dense/BiasAdd_grad/tuple/control_dependency_1Identity+gradients/pi/dense/BiasAdd_grad/BiasAddGrad1^gradients/pi/dense/BiasAdd_grad/tuple/group_deps*
T0*
_output_shapes
:@*>
_class4
20loc:@gradients/pi/dense/BiasAdd_grad/BiasAddGrad
„
%gradients/pi/dense/MatMul_grad/MatMulMatMul8gradients/pi/dense/BiasAdd_grad/tuple/control_dependencypi/dense/kernel/read*'
_output_shapes
:€€€€€€€€€*
T0*
transpose_b(*
transpose_a( 
«
'gradients/pi/dense/MatMul_grad/MatMul_1MatMulPlaceholder8gradients/pi/dense/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
_output_shapes

:@*
transpose_a(*
T0
Й
/gradients/pi/dense/MatMul_grad/tuple/group_depsNoOp&^gradients/pi/dense/MatMul_grad/MatMul(^gradients/pi/dense/MatMul_grad/MatMul_1
И
7gradients/pi/dense/MatMul_grad/tuple/control_dependencyIdentity%gradients/pi/dense/MatMul_grad/MatMul0^gradients/pi/dense/MatMul_grad/tuple/group_deps*8
_class.
,*loc:@gradients/pi/dense/MatMul_grad/MatMul*'
_output_shapes
:€€€€€€€€€*
T0
Е
9gradients/pi/dense/MatMul_grad/tuple/control_dependency_1Identity'gradients/pi/dense/MatMul_grad/MatMul_10^gradients/pi/dense/MatMul_grad/tuple/group_deps*:
_class0
.,loc:@gradients/pi/dense/MatMul_grad/MatMul_1*
T0*
_output_shapes

:@
`
Reshape/shapeConst*
valueB:
€€€€€€€€€*
_output_shapes
:*
dtype0
Р
ReshapeReshape9gradients/pi/dense/MatMul_grad/tuple/control_dependency_1Reshape/shape*
Tshape0*
T0*
_output_shapes	
:А
b
Reshape_1/shapeConst*
valueB:
€€€€€€€€€*
_output_shapes
:*
dtype0
Ф
	Reshape_1Reshape:gradients/pi/dense/BiasAdd_grad/tuple/control_dependency_1Reshape_1/shape*
T0*
Tshape0*
_output_shapes
:@
b
Reshape_2/shapeConst*
dtype0*
valueB:
€€€€€€€€€*
_output_shapes
:
Ц
	Reshape_2Reshape;gradients/pi/dense_1/MatMul_grad/tuple/control_dependency_1Reshape_2/shape*
T0*
_output_shapes	
:А *
Tshape0
b
Reshape_3/shapeConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€
Ц
	Reshape_3Reshape<gradients/pi/dense_1/BiasAdd_grad/tuple/control_dependency_1Reshape_3/shape*
T0*
_output_shapes
:@*
Tshape0
b
Reshape_4/shapeConst*
valueB:
€€€€€€€€€*
dtype0*
_output_shapes
:
Ц
	Reshape_4Reshape;gradients/pi/dense_2/MatMul_grad/tuple/control_dependency_1Reshape_4/shape*
Tshape0*
_output_shapes	
:ј*
T0
b
Reshape_5/shapeConst*
_output_shapes
:*
valueB:
€€€€€€€€€*
dtype0
Ц
	Reshape_5Reshape<gradients/pi/dense_2/BiasAdd_grad/tuple/control_dependency_1Reshape_5/shape*
_output_shapes
:*
T0*
Tshape0
M
concat/axisConst*
dtype0*
_output_shapes
: *
value	B : 
Ъ
concatConcatV2Reshape	Reshape_1	Reshape_2	Reshape_3	Reshape_4	Reshape_5concat/axis*
N*
T0*

Tidx0*
_output_shapes	
:√%
g
PyFuncPyFuncconcat*
Tout
2*
token
pyfunc_0*
Tin
2*
_output_shapes	
:√%
h
Const_5Const*
_output_shapes
:*
dtype0*-
value$B""А  @      @   ј      
Q
split/split_dimConst*
_output_shapes
: *
value	B : *
dtype0
Ф
splitSplitVPyFuncConst_5split/split_dim*
	num_split*;
_output_shapes)
':А:@:А :@:ј:*
T0*

Tlen0
`
Reshape_6/shapeConst*
valueB"   @   *
_output_shapes
:*
dtype0
c
	Reshape_6ReshapesplitReshape_6/shape*
_output_shapes

:@*
T0*
Tshape0
Y
Reshape_7/shapeConst*
_output_shapes
:*
dtype0*
valueB:@
a
	Reshape_7Reshapesplit:1Reshape_7/shape*
Tshape0*
_output_shapes
:@*
T0
`
Reshape_8/shapeConst*
dtype0*
valueB"@   @   *
_output_shapes
:
e
	Reshape_8Reshapesplit:2Reshape_8/shape*
Tshape0*
_output_shapes

:@@*
T0
Y
Reshape_9/shapeConst*
dtype0*
_output_shapes
:*
valueB:@
a
	Reshape_9Reshapesplit:3Reshape_9/shape*
_output_shapes
:@*
T0*
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
Reshape_11/shapeConst*
dtype0*
valueB:*
_output_shapes
:
c

Reshape_11Reshapesplit:5Reshape_11/shape*
_output_shapes
:*
T0*
Tshape0
А
beta1_power/initial_valueConst*
_output_shapes
: *
dtype0*
valueB
 *fff?* 
_class
loc:@pi/dense/bias
С
beta1_power
VariableV2*
	container * 
_class
loc:@pi/dense/bias*
_output_shapes
: *
shape: *
shared_name *
dtype0
∞
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
_output_shapes
: *
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(
l
beta1_power/readIdentitybeta1_power*
_output_shapes
: *
T0* 
_class
loc:@pi/dense/bias
А
beta2_power/initial_valueConst*
valueB
 *wЊ?* 
_class
loc:@pi/dense/bias*
dtype0*
_output_shapes
: 
С
beta2_power
VariableV2*
dtype0*
shared_name * 
_class
loc:@pi/dense/bias*
shape: *
_output_shapes
: *
	container 
∞
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value* 
_class
loc:@pi/dense/bias*
T0*
validate_shape(*
_output_shapes
: *
use_locking(
l
beta2_power/readIdentitybeta2_power*
T0* 
_class
loc:@pi/dense/bias*
_output_shapes
: 
Я
&pi/dense/kernel/Adam/Initializer/zerosConst*
valueB@*    *
dtype0*"
_class
loc:@pi/dense/kernel*
_output_shapes

:@
ђ
pi/dense/kernel/Adam
VariableV2*
shared_name *
shape
:@*
_output_shapes

:@*
dtype0*
	container *"
_class
loc:@pi/dense/kernel
ў
pi/dense/kernel/Adam/AssignAssignpi/dense/kernel/Adam&pi/dense/kernel/Adam/Initializer/zeros*
use_locking(*"
_class
loc:@pi/dense/kernel*
_output_shapes

:@*
T0*
validate_shape(
И
pi/dense/kernel/Adam/readIdentitypi/dense/kernel/Adam*
_output_shapes

:@*
T0*"
_class
loc:@pi/dense/kernel
°
(pi/dense/kernel/Adam_1/Initializer/zerosConst*
valueB@*    *"
_class
loc:@pi/dense/kernel*
_output_shapes

:@*
dtype0
Ѓ
pi/dense/kernel/Adam_1
VariableV2*
shared_name *
dtype0*
shape
:@*
_output_shapes

:@*
	container *"
_class
loc:@pi/dense/kernel
я
pi/dense/kernel/Adam_1/AssignAssignpi/dense/kernel/Adam_1(pi/dense/kernel/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes

:@*
T0*
use_locking(*"
_class
loc:@pi/dense/kernel
М
pi/dense/kernel/Adam_1/readIdentitypi/dense/kernel/Adam_1*"
_class
loc:@pi/dense/kernel*
_output_shapes

:@*
T0
У
$pi/dense/bias/Adam/Initializer/zerosConst* 
_class
loc:@pi/dense/bias*
valueB@*    *
dtype0*
_output_shapes
:@
†
pi/dense/bias/Adam
VariableV2*
shape:@*
shared_name *
	container *
dtype0*
_output_shapes
:@* 
_class
loc:@pi/dense/bias
Ќ
pi/dense/bias/Adam/AssignAssignpi/dense/bias/Adam$pi/dense/bias/Adam/Initializer/zeros*
use_locking(* 
_class
loc:@pi/dense/bias*
T0*
validate_shape(*
_output_shapes
:@
~
pi/dense/bias/Adam/readIdentitypi/dense/bias/Adam* 
_class
loc:@pi/dense/bias*
_output_shapes
:@*
T0
Х
&pi/dense/bias/Adam_1/Initializer/zerosConst*
_output_shapes
:@* 
_class
loc:@pi/dense/bias*
dtype0*
valueB@*    
Ґ
pi/dense/bias/Adam_1
VariableV2*
_output_shapes
:@* 
_class
loc:@pi/dense/bias*
	container *
shared_name *
shape:@*
dtype0
”
pi/dense/bias/Adam_1/AssignAssignpi/dense/bias/Adam_1&pi/dense/bias/Adam_1/Initializer/zeros*
validate_shape(*
use_locking(* 
_class
loc:@pi/dense/bias*
T0*
_output_shapes
:@
В
pi/dense/bias/Adam_1/readIdentitypi/dense/bias/Adam_1* 
_class
loc:@pi/dense/bias*
T0*
_output_shapes
:@
ѓ
8pi/dense_1/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
valueB"@   @   *
_output_shapes
:*
dtype0*$
_class
loc:@pi/dense_1/kernel
Щ
.pi/dense_1/kernel/Adam/Initializer/zeros/ConstConst*$
_class
loc:@pi/dense_1/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
ы
(pi/dense_1/kernel/Adam/Initializer/zerosFill8pi/dense_1/kernel/Adam/Initializer/zeros/shape_as_tensor.pi/dense_1/kernel/Adam/Initializer/zeros/Const*

index_type0*
T0*
_output_shapes

:@@*$
_class
loc:@pi/dense_1/kernel
∞
pi/dense_1/kernel/Adam
VariableV2*
shape
:@@*
	container *
dtype0*
shared_name *$
_class
loc:@pi/dense_1/kernel*
_output_shapes

:@@
б
pi/dense_1/kernel/Adam/AssignAssignpi/dense_1/kernel/Adam(pi/dense_1/kernel/Adam/Initializer/zeros*
_output_shapes

:@@*
T0*
validate_shape(*$
_class
loc:@pi/dense_1/kernel*
use_locking(
О
pi/dense_1/kernel/Adam/readIdentitypi/dense_1/kernel/Adam*$
_class
loc:@pi/dense_1/kernel*
T0*
_output_shapes

:@@
±
:pi/dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*$
_class
loc:@pi/dense_1/kernel*
valueB"@   @   
Ы
0pi/dense_1/kernel/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: *$
_class
loc:@pi/dense_1/kernel
Б
*pi/dense_1/kernel/Adam_1/Initializer/zerosFill:pi/dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensor0pi/dense_1/kernel/Adam_1/Initializer/zeros/Const*
T0*
_output_shapes

:@@*$
_class
loc:@pi/dense_1/kernel*

index_type0
≤
pi/dense_1/kernel/Adam_1
VariableV2*$
_class
loc:@pi/dense_1/kernel*
dtype0*
	container *
_output_shapes

:@@*
shape
:@@*
shared_name 
з
pi/dense_1/kernel/Adam_1/AssignAssignpi/dense_1/kernel/Adam_1*pi/dense_1/kernel/Adam_1/Initializer/zeros*
T0*
_output_shapes

:@@*
validate_shape(*
use_locking(*$
_class
loc:@pi/dense_1/kernel
Т
pi/dense_1/kernel/Adam_1/readIdentitypi/dense_1/kernel/Adam_1*
_output_shapes

:@@*$
_class
loc:@pi/dense_1/kernel*
T0
Ч
&pi/dense_1/bias/Adam/Initializer/zerosConst*
valueB@*    *
_output_shapes
:@*"
_class
loc:@pi/dense_1/bias*
dtype0
§
pi/dense_1/bias/Adam
VariableV2*
dtype0*
_output_shapes
:@*
shape:@*
shared_name *
	container *"
_class
loc:@pi/dense_1/bias
’
pi/dense_1/bias/Adam/AssignAssignpi/dense_1/bias/Adam&pi/dense_1/bias/Adam/Initializer/zeros*"
_class
loc:@pi/dense_1/bias*
_output_shapes
:@*
use_locking(*
T0*
validate_shape(
Д
pi/dense_1/bias/Adam/readIdentitypi/dense_1/bias/Adam*
_output_shapes
:@*"
_class
loc:@pi/dense_1/bias*
T0
Щ
(pi/dense_1/bias/Adam_1/Initializer/zerosConst*
valueB@*    *"
_class
loc:@pi/dense_1/bias*
_output_shapes
:@*
dtype0
¶
pi/dense_1/bias/Adam_1
VariableV2*"
_class
loc:@pi/dense_1/bias*
	container *
_output_shapes
:@*
dtype0*
shape:@*
shared_name 
џ
pi/dense_1/bias/Adam_1/AssignAssignpi/dense_1/bias/Adam_1(pi/dense_1/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*
_output_shapes
:@*
validate_shape(*"
_class
loc:@pi/dense_1/bias
И
pi/dense_1/bias/Adam_1/readIdentitypi/dense_1/bias/Adam_1*
T0*"
_class
loc:@pi/dense_1/bias*
_output_shapes
:@
£
(pi/dense_2/kernel/Adam/Initializer/zerosConst*
_output_shapes

:@*$
_class
loc:@pi/dense_2/kernel*
valueB@*    *
dtype0
∞
pi/dense_2/kernel/Adam
VariableV2*
shared_name *
shape
:@*
	container *$
_class
loc:@pi/dense_2/kernel*
dtype0*
_output_shapes

:@
б
pi/dense_2/kernel/Adam/AssignAssignpi/dense_2/kernel/Adam(pi/dense_2/kernel/Adam/Initializer/zeros*
validate_shape(*
T0*
_output_shapes

:@*$
_class
loc:@pi/dense_2/kernel*
use_locking(
О
pi/dense_2/kernel/Adam/readIdentitypi/dense_2/kernel/Adam*
T0*
_output_shapes

:@*$
_class
loc:@pi/dense_2/kernel
•
*pi/dense_2/kernel/Adam_1/Initializer/zerosConst*$
_class
loc:@pi/dense_2/kernel*
valueB@*    *
_output_shapes

:@*
dtype0
≤
pi/dense_2/kernel/Adam_1
VariableV2*
shape
:@*
shared_name *
_output_shapes

:@*
	container *$
_class
loc:@pi/dense_2/kernel*
dtype0
з
pi/dense_2/kernel/Adam_1/AssignAssignpi/dense_2/kernel/Adam_1*pi/dense_2/kernel/Adam_1/Initializer/zeros*
use_locking(*
_output_shapes

:@*
validate_shape(*$
_class
loc:@pi/dense_2/kernel*
T0
Т
pi/dense_2/kernel/Adam_1/readIdentitypi/dense_2/kernel/Adam_1*
_output_shapes

:@*$
_class
loc:@pi/dense_2/kernel*
T0
Ч
&pi/dense_2/bias/Adam/Initializer/zerosConst*
valueB*    *
dtype0*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:
§
pi/dense_2/bias/Adam
VariableV2*
dtype0*
	container *
shared_name *
shape:*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:
’
pi/dense_2/bias/Adam/AssignAssignpi/dense_2/bias/Adam&pi/dense_2/bias/Adam/Initializer/zeros*
_output_shapes
:*
use_locking(*
validate_shape(*
T0*"
_class
loc:@pi/dense_2/bias
Д
pi/dense_2/bias/Adam/readIdentitypi/dense_2/bias/Adam*
T0*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:
Щ
(pi/dense_2/bias/Adam_1/Initializer/zerosConst*
dtype0*
valueB*    *
_output_shapes
:*"
_class
loc:@pi/dense_2/bias
¶
pi/dense_2/bias/Adam_1
VariableV2*
_output_shapes
:*
	container *
shared_name *"
_class
loc:@pi/dense_2/bias*
dtype0*
shape:
џ
pi/dense_2/bias/Adam_1/AssignAssignpi/dense_2/bias/Adam_1(pi/dense_2/bias/Adam_1/Initializer/zeros*
_output_shapes
:*
use_locking(*"
_class
loc:@pi/dense_2/bias*
T0*
validate_shape(
И
pi/dense_2/bias/Adam_1/readIdentitypi/dense_2/bias/Adam_1*"
_class
loc:@pi/dense_2/bias*
T0*
_output_shapes
:
W
Adam/learning_rateConst*
valueB
 *RIЭ9*
_output_shapes
: *
dtype0
O

Adam/beta1Const*
dtype0*
_output_shapes
: *
valueB
 *fff?
O

Adam/beta2Const*
_output_shapes
: *
valueB
 *wЊ?*
dtype0
Q
Adam/epsilonConst*
_output_shapes
: *
dtype0*
valueB
 *wћ+2
ќ
%Adam/update_pi/dense/kernel/ApplyAdam	ApplyAdampi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon	Reshape_6*
_output_shapes

:@*
use_locking( *
use_nesterov( *
T0*"
_class
loc:@pi/dense/kernel
ј
#Adam/update_pi/dense/bias/ApplyAdam	ApplyAdampi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon	Reshape_7* 
_class
loc:@pi/dense/bias*
T0*
use_locking( *
use_nesterov( *
_output_shapes
:@
Ў
'Adam/update_pi/dense_1/kernel/ApplyAdam	ApplyAdampi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon	Reshape_8*
T0*
use_nesterov( *$
_class
loc:@pi/dense_1/kernel*
use_locking( *
_output_shapes

:@@
 
%Adam/update_pi/dense_1/bias/ApplyAdam	ApplyAdampi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon	Reshape_9*
T0*
use_locking( *"
_class
loc:@pi/dense_1/bias*
_output_shapes
:@*
use_nesterov( 
ў
'Adam/update_pi/dense_2/kernel/ApplyAdam	ApplyAdampi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon
Reshape_10*
_output_shapes

:@*
use_locking( *
use_nesterov( *$
_class
loc:@pi/dense_2/kernel*
T0
Ћ
%Adam/update_pi/dense_2/bias/ApplyAdam	ApplyAdampi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon
Reshape_11*
T0*"
_class
loc:@pi/dense_2/bias*
use_locking( *
_output_shapes
:*
use_nesterov( 
в
Adam/mulMulbeta1_power/read
Adam/beta1$^Adam/update_pi/dense/bias/ApplyAdam&^Adam/update_pi/dense/kernel/ApplyAdam&^Adam/update_pi/dense_1/bias/ApplyAdam(^Adam/update_pi/dense_1/kernel/ApplyAdam&^Adam/update_pi/dense_2/bias/ApplyAdam(^Adam/update_pi/dense_2/kernel/ApplyAdam*
T0* 
_class
loc:@pi/dense/bias*
_output_shapes
: 
Ш
Adam/AssignAssignbeta1_powerAdam/mul* 
_class
loc:@pi/dense/bias*
_output_shapes
: *
T0*
use_locking( *
validate_shape(
д

Adam/mul_1Mulbeta2_power/read
Adam/beta2$^Adam/update_pi/dense/bias/ApplyAdam&^Adam/update_pi/dense/kernel/ApplyAdam&^Adam/update_pi/dense_1/bias/ApplyAdam(^Adam/update_pi/dense_1/kernel/ApplyAdam&^Adam/update_pi/dense_2/bias/ApplyAdam(^Adam/update_pi/dense_2/kernel/ApplyAdam*
_output_shapes
: *
T0* 
_class
loc:@pi/dense/bias
Ь
Adam/Assign_1Assignbeta2_power
Adam/mul_1*
validate_shape(*
_output_shapes
: *
use_locking( * 
_class
loc:@pi/dense/bias*
T0
Ь
AdamNoOp^Adam/Assign^Adam/Assign_1$^Adam/update_pi/dense/bias/ApplyAdam&^Adam/update_pi/dense/kernel/ApplyAdam&^Adam/update_pi/dense_1/bias/ApplyAdam(^Adam/update_pi/dense_1/kernel/ApplyAdam&^Adam/update_pi/dense_2/bias/ApplyAdam(^Adam/update_pi/dense_2/kernel/ApplyAdam
j
Reshape_12/shapeConst^Adam*
valueB:
€€€€€€€€€*
_output_shapes
:*
dtype0
q

Reshape_12Reshapepi/dense/kernel/readReshape_12/shape*
Tshape0*
_output_shapes	
:А*
T0
j
Reshape_13/shapeConst^Adam*
dtype0*
_output_shapes
:*
valueB:
€€€€€€€€€
n

Reshape_13Reshapepi/dense/bias/readReshape_13/shape*
_output_shapes
:@*
Tshape0*
T0
j
Reshape_14/shapeConst^Adam*
dtype0*
valueB:
€€€€€€€€€*
_output_shapes
:
s

Reshape_14Reshapepi/dense_1/kernel/readReshape_14/shape*
Tshape0*
T0*
_output_shapes	
:А 
j
Reshape_15/shapeConst^Adam*
valueB:
€€€€€€€€€*
dtype0*
_output_shapes
:
p

Reshape_15Reshapepi/dense_1/bias/readReshape_15/shape*
Tshape0*
T0*
_output_shapes
:@
j
Reshape_16/shapeConst^Adam*
_output_shapes
:*
valueB:
€€€€€€€€€*
dtype0
s

Reshape_16Reshapepi/dense_2/kernel/readReshape_16/shape*
Tshape0*
T0*
_output_shapes	
:ј
j
Reshape_17/shapeConst^Adam*
dtype0*
_output_shapes
:*
valueB:
€€€€€€€€€
p

Reshape_17Reshapepi/dense_2/bias/readReshape_17/shape*
_output_shapes
:*
Tshape0*
T0
V
concat_1/axisConst^Adam*
value	B : *
_output_shapes
: *
dtype0
¶
concat_1ConcatV2
Reshape_12
Reshape_13
Reshape_14
Reshape_15
Reshape_16
Reshape_17concat_1/axis*

Tidx0*
T0*
N*
_output_shapes	
:√%
h
PyFunc_1PyFuncconcat_1*
token
pyfunc_1*
_output_shapes
:*
Tout
2*
Tin
2
o
Const_6Const^Adam*
dtype0*
_output_shapes
:*-
value$B""А  @      @   ј      
Z
split_1/split_dimConst^Adam*
dtype0*
value	B : *
_output_shapes
: 
Л
split_1SplitVPyFunc_1Const_6split_1/split_dim*

Tlen0*
T0*,
_output_shapes
::::::*
	num_split
h
Reshape_18/shapeConst^Adam*
_output_shapes
:*
valueB"   @   *
dtype0
g

Reshape_18Reshapesplit_1Reshape_18/shape*
T0*
Tshape0*
_output_shapes

:@
a
Reshape_19/shapeConst^Adam*
valueB:@*
dtype0*
_output_shapes
:
e

Reshape_19Reshape	split_1:1Reshape_19/shape*
T0*
Tshape0*
_output_shapes
:@
h
Reshape_20/shapeConst^Adam*
valueB"@   @   *
dtype0*
_output_shapes
:
i

Reshape_20Reshape	split_1:2Reshape_20/shape*
Tshape0*
_output_shapes

:@@*
T0
a
Reshape_21/shapeConst^Adam*
dtype0*
_output_shapes
:*
valueB:@
e

Reshape_21Reshape	split_1:3Reshape_21/shape*
_output_shapes
:@*
T0*
Tshape0
h
Reshape_22/shapeConst^Adam*
valueB"@      *
dtype0*
_output_shapes
:
i

Reshape_22Reshape	split_1:4Reshape_22/shape*
Tshape0*
_output_shapes

:@*
T0
a
Reshape_23/shapeConst^Adam*
_output_shapes
:*
dtype0*
valueB:
e

Reshape_23Reshape	split_1:5Reshape_23/shape*
Tshape0*
_output_shapes
:*
T0
£
AssignAssignpi/dense/kernel
Reshape_18*
T0*
use_locking(*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes

:@
Э
Assign_1Assignpi/dense/bias
Reshape_19*
T0*
use_locking(*
_output_shapes
:@*
validate_shape(* 
_class
loc:@pi/dense/bias
©
Assign_2Assignpi/dense_1/kernel
Reshape_20*$
_class
loc:@pi/dense_1/kernel*
_output_shapes

:@@*
validate_shape(*
use_locking(*
T0
°
Assign_3Assignpi/dense_1/bias
Reshape_21*
T0*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
use_locking(*
_output_shapes
:@
©
Assign_4Assignpi/dense_2/kernel
Reshape_22*
validate_shape(*$
_class
loc:@pi/dense_2/kernel*
use_locking(*
_output_shapes

:@*
T0
°
Assign_5Assignpi/dense_2/bias
Reshape_23*
T0*
_output_shapes
:*
validate_shape(*
use_locking(*"
_class
loc:@pi/dense_2/bias
Y

group_depsNoOp^Adam^Assign	^Assign_1	^Assign_2	^Assign_3	^Assign_4	^Assign_5
(
group_deps_1NoOp^Adam^group_deps
T
gradients_1/ShapeConst*
dtype0*
valueB *
_output_shapes
: 
Z
gradients_1/grad_ys_0Const*
_output_shapes
: *
dtype0*
valueB
 *  А?
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
Ц
gradients_1/Mean_1_grad/ReshapeReshapegradients_1/Fill%gradients_1/Mean_1_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
:
`
gradients_1/Mean_1_grad/ShapeShapepow*
_output_shapes
:*
out_type0*
T0
§
gradients_1/Mean_1_grad/TileTilegradients_1/Mean_1_grad/Reshapegradients_1/Mean_1_grad/Shape*

Tmultiples0*#
_output_shapes
:€€€€€€€€€*
T0
b
gradients_1/Mean_1_grad/Shape_1Shapepow*
T0*
_output_shapes
:*
out_type0
b
gradients_1/Mean_1_grad/Shape_2Const*
valueB *
_output_shapes
: *
dtype0
g
gradients_1/Mean_1_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
Ґ
gradients_1/Mean_1_grad/ProdProdgradients_1/Mean_1_grad/Shape_1gradients_1/Mean_1_grad/Const*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
i
gradients_1/Mean_1_grad/Const_1Const*
dtype0*
valueB: *
_output_shapes
:
¶
gradients_1/Mean_1_grad/Prod_1Prodgradients_1/Mean_1_grad/Shape_2gradients_1/Mean_1_grad/Const_1*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
c
!gradients_1/Mean_1_grad/Maximum/yConst*
_output_shapes
: *
dtype0*
value	B :
О
gradients_1/Mean_1_grad/MaximumMaximumgradients_1/Mean_1_grad/Prod_1!gradients_1/Mean_1_grad/Maximum/y*
T0*
_output_shapes
: 
М
 gradients_1/Mean_1_grad/floordivFloorDivgradients_1/Mean_1_grad/Prodgradients_1/Mean_1_grad/Maximum*
T0*
_output_shapes
: 
Ж
gradients_1/Mean_1_grad/CastCast gradients_1/Mean_1_grad/floordiv*
Truncate( *
_output_shapes
: *

DstT0*

SrcT0
Ф
gradients_1/Mean_1_grad/truedivRealDivgradients_1/Mean_1_grad/Tilegradients_1/Mean_1_grad/Cast*
T0*#
_output_shapes
:€€€€€€€€€
_
gradients_1/pow_grad/ShapeShapesub_1*
_output_shapes
:*
T0*
out_type0
_
gradients_1/pow_grad/Shape_1Const*
valueB *
_output_shapes
: *
dtype0
Ї
*gradients_1/pow_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/pow_grad/Shapegradients_1/pow_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
u
gradients_1/pow_grad/mulMulgradients_1/Mean_1_grad/truedivpow/y*#
_output_shapes
:€€€€€€€€€*
T0
_
gradients_1/pow_grad/sub/yConst*
dtype0*
_output_shapes
: *
valueB
 *  А?
c
gradients_1/pow_grad/subSubpow/ygradients_1/pow_grad/sub/y*
_output_shapes
: *
T0
n
gradients_1/pow_grad/PowPowsub_1gradients_1/pow_grad/sub*#
_output_shapes
:€€€€€€€€€*
T0
Г
gradients_1/pow_grad/mul_1Mulgradients_1/pow_grad/mulgradients_1/pow_grad/Pow*#
_output_shapes
:€€€€€€€€€*
T0
І
gradients_1/pow_grad/SumSumgradients_1/pow_grad/mul_1*gradients_1/pow_grad/BroadcastGradientArgs*
T0*
	keep_dims( *

Tidx0*
_output_shapes
:
Щ
gradients_1/pow_grad/ReshapeReshapegradients_1/pow_grad/Sumgradients_1/pow_grad/Shape*#
_output_shapes
:€€€€€€€€€*
Tshape0*
T0
c
gradients_1/pow_grad/Greater/yConst*
_output_shapes
: *
valueB
 *    *
dtype0
|
gradients_1/pow_grad/GreaterGreatersub_1gradients_1/pow_grad/Greater/y*#
_output_shapes
:€€€€€€€€€*
T0
i
$gradients_1/pow_grad/ones_like/ShapeShapesub_1*
out_type0*
_output_shapes
:*
T0
i
$gradients_1/pow_grad/ones_like/ConstConst*
_output_shapes
: *
valueB
 *  А?*
dtype0
≤
gradients_1/pow_grad/ones_likeFill$gradients_1/pow_grad/ones_like/Shape$gradients_1/pow_grad/ones_like/Const*#
_output_shapes
:€€€€€€€€€*
T0*

index_type0
Ш
gradients_1/pow_grad/SelectSelectgradients_1/pow_grad/Greatersub_1gradients_1/pow_grad/ones_like*#
_output_shapes
:€€€€€€€€€*
T0
j
gradients_1/pow_grad/LogLoggradients_1/pow_grad/Select*
T0*#
_output_shapes
:€€€€€€€€€
a
gradients_1/pow_grad/zeros_like	ZerosLikesub_1*#
_output_shapes
:€€€€€€€€€*
T0
Ѓ
gradients_1/pow_grad/Select_1Selectgradients_1/pow_grad/Greatergradients_1/pow_grad/Loggradients_1/pow_grad/zeros_like*
T0*#
_output_shapes
:€€€€€€€€€
u
gradients_1/pow_grad/mul_2Mulgradients_1/Mean_1_grad/truedivpow*#
_output_shapes
:€€€€€€€€€*
T0
К
gradients_1/pow_grad/mul_3Mulgradients_1/pow_grad/mul_2gradients_1/pow_grad/Select_1*#
_output_shapes
:€€€€€€€€€*
T0
Ђ
gradients_1/pow_grad/Sum_1Sumgradients_1/pow_grad/mul_3,gradients_1/pow_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
Т
gradients_1/pow_grad/Reshape_1Reshapegradients_1/pow_grad/Sum_1gradients_1/pow_grad/Shape_1*
Tshape0*
T0*
_output_shapes
: 
m
%gradients_1/pow_grad/tuple/group_depsNoOp^gradients_1/pow_grad/Reshape^gradients_1/pow_grad/Reshape_1
ё
-gradients_1/pow_grad/tuple/control_dependencyIdentitygradients_1/pow_grad/Reshape&^gradients_1/pow_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients_1/pow_grad/Reshape*#
_output_shapes
:€€€€€€€€€
„
/gradients_1/pow_grad/tuple/control_dependency_1Identitygradients_1/pow_grad/Reshape_1&^gradients_1/pow_grad/tuple/group_deps*1
_class'
%#loc:@gradients_1/pow_grad/Reshape_1*
T0*
_output_shapes
: 
i
gradients_1/sub_1_grad/ShapeShapePlaceholder_3*
_output_shapes
:*
out_type0*
T0
g
gradients_1/sub_1_grad/Shape_1Shape	v/Squeeze*
out_type0*
_output_shapes
:*
T0
ј
,gradients_1/sub_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/sub_1_grad/Shapegradients_1/sub_1_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
Њ
gradients_1/sub_1_grad/SumSum-gradients_1/pow_grad/tuple/control_dependency,gradients_1/sub_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
_output_shapes
:*
T0
Я
gradients_1/sub_1_grad/ReshapeReshapegradients_1/sub_1_grad/Sumgradients_1/sub_1_grad/Shape*
Tshape0*#
_output_shapes
:€€€€€€€€€*
T0
¬
gradients_1/sub_1_grad/Sum_1Sum-gradients_1/pow_grad/tuple/control_dependency.gradients_1/sub_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
b
gradients_1/sub_1_grad/NegNeggradients_1/sub_1_grad/Sum_1*
_output_shapes
:*
T0
£
 gradients_1/sub_1_grad/Reshape_1Reshapegradients_1/sub_1_grad/Neggradients_1/sub_1_grad/Shape_1*
Tshape0*
T0*#
_output_shapes
:€€€€€€€€€
s
'gradients_1/sub_1_grad/tuple/group_depsNoOp^gradients_1/sub_1_grad/Reshape!^gradients_1/sub_1_grad/Reshape_1
ж
/gradients_1/sub_1_grad/tuple/control_dependencyIdentitygradients_1/sub_1_grad/Reshape(^gradients_1/sub_1_grad/tuple/group_deps*#
_output_shapes
:€€€€€€€€€*1
_class'
%#loc:@gradients_1/sub_1_grad/Reshape*
T0
м
1gradients_1/sub_1_grad/tuple/control_dependency_1Identity gradients_1/sub_1_grad/Reshape_1(^gradients_1/sub_1_grad/tuple/group_deps*#
_output_shapes
:€€€€€€€€€*3
_class)
'%loc:@gradients_1/sub_1_grad/Reshape_1*
T0
q
 gradients_1/v/Squeeze_grad/ShapeShapev/dense_2/BiasAdd*
T0*
out_type0*
_output_shapes
:
¬
"gradients_1/v/Squeeze_grad/ReshapeReshape1gradients_1/sub_1_grad/tuple/control_dependency_1 gradients_1/v/Squeeze_grad/Shape*'
_output_shapes
:€€€€€€€€€*
T0*
Tshape0
Э
.gradients_1/v/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad"gradients_1/v/Squeeze_grad/Reshape*
_output_shapes
:*
data_formatNHWC*
T0
С
3gradients_1/v/dense_2/BiasAdd_grad/tuple/group_depsNoOp#^gradients_1/v/Squeeze_grad/Reshape/^gradients_1/v/dense_2/BiasAdd_grad/BiasAddGrad
К
;gradients_1/v/dense_2/BiasAdd_grad/tuple/control_dependencyIdentity"gradients_1/v/Squeeze_grad/Reshape4^gradients_1/v/dense_2/BiasAdd_grad/tuple/group_deps*'
_output_shapes
:€€€€€€€€€*5
_class+
)'loc:@gradients_1/v/Squeeze_grad/Reshape*
T0
Ч
=gradients_1/v/dense_2/BiasAdd_grad/tuple/control_dependency_1Identity.gradients_1/v/dense_2/BiasAdd_grad/BiasAddGrad4^gradients_1/v/dense_2/BiasAdd_grad/tuple/group_deps*
_output_shapes
:*
T0*A
_class7
53loc:@gradients_1/v/dense_2/BiasAdd_grad/BiasAddGrad
ё
(gradients_1/v/dense_2/MatMul_grad/MatMulMatMul;gradients_1/v/dense_2/BiasAdd_grad/tuple/control_dependencyv/dense_2/kernel/read*
T0*'
_output_shapes
:€€€€€€€€€@*
transpose_b(*
transpose_a( 
–
*gradients_1/v/dense_2/MatMul_grad/MatMul_1MatMulv/dense_1/Tanh;gradients_1/v/dense_2/BiasAdd_grad/tuple/control_dependency*
transpose_a(*
T0*
transpose_b( *
_output_shapes

:@
Т
2gradients_1/v/dense_2/MatMul_grad/tuple/group_depsNoOp)^gradients_1/v/dense_2/MatMul_grad/MatMul+^gradients_1/v/dense_2/MatMul_grad/MatMul_1
Ф
:gradients_1/v/dense_2/MatMul_grad/tuple/control_dependencyIdentity(gradients_1/v/dense_2/MatMul_grad/MatMul3^gradients_1/v/dense_2/MatMul_grad/tuple/group_deps*;
_class1
/-loc:@gradients_1/v/dense_2/MatMul_grad/MatMul*'
_output_shapes
:€€€€€€€€€@*
T0
С
<gradients_1/v/dense_2/MatMul_grad/tuple/control_dependency_1Identity*gradients_1/v/dense_2/MatMul_grad/MatMul_13^gradients_1/v/dense_2/MatMul_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients_1/v/dense_2/MatMul_grad/MatMul_1*
_output_shapes

:@
≤
(gradients_1/v/dense_1/Tanh_grad/TanhGradTanhGradv/dense_1/Tanh:gradients_1/v/dense_2/MatMul_grad/tuple/control_dependency*'
_output_shapes
:€€€€€€€€€@*
T0
£
.gradients_1/v/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad(gradients_1/v/dense_1/Tanh_grad/TanhGrad*
data_formatNHWC*
T0*
_output_shapes
:@
Ч
3gradients_1/v/dense_1/BiasAdd_grad/tuple/group_depsNoOp/^gradients_1/v/dense_1/BiasAdd_grad/BiasAddGrad)^gradients_1/v/dense_1/Tanh_grad/TanhGrad
Ц
;gradients_1/v/dense_1/BiasAdd_grad/tuple/control_dependencyIdentity(gradients_1/v/dense_1/Tanh_grad/TanhGrad4^gradients_1/v/dense_1/BiasAdd_grad/tuple/group_deps*
T0*'
_output_shapes
:€€€€€€€€€@*;
_class1
/-loc:@gradients_1/v/dense_1/Tanh_grad/TanhGrad
Ч
=gradients_1/v/dense_1/BiasAdd_grad/tuple/control_dependency_1Identity.gradients_1/v/dense_1/BiasAdd_grad/BiasAddGrad4^gradients_1/v/dense_1/BiasAdd_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients_1/v/dense_1/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@
ё
(gradients_1/v/dense_1/MatMul_grad/MatMulMatMul;gradients_1/v/dense_1/BiasAdd_grad/tuple/control_dependencyv/dense_1/kernel/read*
transpose_a( *
T0*
transpose_b(*'
_output_shapes
:€€€€€€€€€@
ќ
*gradients_1/v/dense_1/MatMul_grad/MatMul_1MatMulv/dense/Tanh;gradients_1/v/dense_1/BiasAdd_grad/tuple/control_dependency*
transpose_a(*
_output_shapes

:@@*
transpose_b( *
T0
Т
2gradients_1/v/dense_1/MatMul_grad/tuple/group_depsNoOp)^gradients_1/v/dense_1/MatMul_grad/MatMul+^gradients_1/v/dense_1/MatMul_grad/MatMul_1
Ф
:gradients_1/v/dense_1/MatMul_grad/tuple/control_dependencyIdentity(gradients_1/v/dense_1/MatMul_grad/MatMul3^gradients_1/v/dense_1/MatMul_grad/tuple/group_deps*'
_output_shapes
:€€€€€€€€€@*;
_class1
/-loc:@gradients_1/v/dense_1/MatMul_grad/MatMul*
T0
С
<gradients_1/v/dense_1/MatMul_grad/tuple/control_dependency_1Identity*gradients_1/v/dense_1/MatMul_grad/MatMul_13^gradients_1/v/dense_1/MatMul_grad/tuple/group_deps*
T0*
_output_shapes

:@@*=
_class3
1/loc:@gradients_1/v/dense_1/MatMul_grad/MatMul_1
Ѓ
&gradients_1/v/dense/Tanh_grad/TanhGradTanhGradv/dense/Tanh:gradients_1/v/dense_1/MatMul_grad/tuple/control_dependency*'
_output_shapes
:€€€€€€€€€@*
T0
Я
,gradients_1/v/dense/BiasAdd_grad/BiasAddGradBiasAddGrad&gradients_1/v/dense/Tanh_grad/TanhGrad*
T0*
data_formatNHWC*
_output_shapes
:@
С
1gradients_1/v/dense/BiasAdd_grad/tuple/group_depsNoOp-^gradients_1/v/dense/BiasAdd_grad/BiasAddGrad'^gradients_1/v/dense/Tanh_grad/TanhGrad
О
9gradients_1/v/dense/BiasAdd_grad/tuple/control_dependencyIdentity&gradients_1/v/dense/Tanh_grad/TanhGrad2^gradients_1/v/dense/BiasAdd_grad/tuple/group_deps*'
_output_shapes
:€€€€€€€€€@*
T0*9
_class/
-+loc:@gradients_1/v/dense/Tanh_grad/TanhGrad
П
;gradients_1/v/dense/BiasAdd_grad/tuple/control_dependency_1Identity,gradients_1/v/dense/BiasAdd_grad/BiasAddGrad2^gradients_1/v/dense/BiasAdd_grad/tuple/group_deps*?
_class5
31loc:@gradients_1/v/dense/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@*
T0
Ў
&gradients_1/v/dense/MatMul_grad/MatMulMatMul9gradients_1/v/dense/BiasAdd_grad/tuple/control_dependencyv/dense/kernel/read*
transpose_b(*'
_output_shapes
:€€€€€€€€€*
T0*
transpose_a( 
…
(gradients_1/v/dense/MatMul_grad/MatMul_1MatMulPlaceholder9gradients_1/v/dense/BiasAdd_grad/tuple/control_dependency*
transpose_a(*
transpose_b( *
_output_shapes

:@*
T0
М
0gradients_1/v/dense/MatMul_grad/tuple/group_depsNoOp'^gradients_1/v/dense/MatMul_grad/MatMul)^gradients_1/v/dense/MatMul_grad/MatMul_1
М
8gradients_1/v/dense/MatMul_grad/tuple/control_dependencyIdentity&gradients_1/v/dense/MatMul_grad/MatMul1^gradients_1/v/dense/MatMul_grad/tuple/group_deps*
T0*9
_class/
-+loc:@gradients_1/v/dense/MatMul_grad/MatMul*'
_output_shapes
:€€€€€€€€€
Й
:gradients_1/v/dense/MatMul_grad/tuple/control_dependency_1Identity(gradients_1/v/dense/MatMul_grad/MatMul_11^gradients_1/v/dense/MatMul_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients_1/v/dense/MatMul_grad/MatMul_1*
_output_shapes

:@
c
Reshape_24/shapeConst*
dtype0*
valueB:
€€€€€€€€€*
_output_shapes
:
Ч

Reshape_24Reshape:gradients_1/v/dense/MatMul_grad/tuple/control_dependency_1Reshape_24/shape*
Tshape0*
T0*
_output_shapes	
:А
c
Reshape_25/shapeConst*
dtype0*
_output_shapes
:*
valueB:
€€€€€€€€€
Ч

Reshape_25Reshape;gradients_1/v/dense/BiasAdd_grad/tuple/control_dependency_1Reshape_25/shape*
_output_shapes
:@*
Tshape0*
T0
c
Reshape_26/shapeConst*
valueB:
€€€€€€€€€*
dtype0*
_output_shapes
:
Щ

Reshape_26Reshape<gradients_1/v/dense_1/MatMul_grad/tuple/control_dependency_1Reshape_26/shape*
Tshape0*
T0*
_output_shapes	
:А 
c
Reshape_27/shapeConst*
dtype0*
valueB:
€€€€€€€€€*
_output_shapes
:
Щ

Reshape_27Reshape=gradients_1/v/dense_1/BiasAdd_grad/tuple/control_dependency_1Reshape_27/shape*
Tshape0*
T0*
_output_shapes
:@
c
Reshape_28/shapeConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€
Ш

Reshape_28Reshape<gradients_1/v/dense_2/MatMul_grad/tuple/control_dependency_1Reshape_28/shape*
Tshape0*
T0*
_output_shapes
:@
c
Reshape_29/shapeConst*
dtype0*
_output_shapes
:*
valueB:
€€€€€€€€€
Щ

Reshape_29Reshape=gradients_1/v/dense_2/BiasAdd_grad/tuple/control_dependency_1Reshape_29/shape*
_output_shapes
:*
Tshape0*
T0
O
concat_2/axisConst*
dtype0*
value	B : *
_output_shapes
: 
¶
concat_2ConcatV2
Reshape_24
Reshape_25
Reshape_26
Reshape_27
Reshape_28
Reshape_29concat_2/axis*

Tidx0*
N*
T0*
_output_shapes	
:Ѕ$
k
PyFunc_2PyFuncconcat_2*
Tout
2*
_output_shapes	
:Ѕ$*
token
pyfunc_2*
Tin
2
h
Const_7Const*
dtype0*
_output_shapes
:*-
value$B""А  @      @   @      
S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
Щ
split_2SplitVPyFunc_2Const_7split_2/split_dim*
T0*

Tlen0*
	num_split*:
_output_shapes(
&:А:@:А :@:@:
a
Reshape_30/shapeConst*
valueB"   @   *
_output_shapes
:*
dtype0
g

Reshape_30Reshapesplit_2Reshape_30/shape*
T0*
_output_shapes

:@*
Tshape0
Z
Reshape_31/shapeConst*
dtype0*
_output_shapes
:*
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

Reshape_32Reshape	split_2:2Reshape_32/shape*
Tshape0*
_output_shapes

:@@*
T0
Z
Reshape_33/shapeConst*
_output_shapes
:*
dtype0*
valueB:@
e

Reshape_33Reshape	split_2:3Reshape_33/shape*
_output_shapes
:@*
Tshape0*
T0
a
Reshape_34/shapeConst*
_output_shapes
:*
dtype0*
valueB"@      
i

Reshape_34Reshape	split_2:4Reshape_34/shape*
_output_shapes

:@*
Tshape0*
T0
Z
Reshape_35/shapeConst*
_output_shapes
:*
valueB:*
dtype0
e

Reshape_35Reshape	split_2:5Reshape_35/shape*
T0*
_output_shapes
:*
Tshape0
Б
beta1_power_1/initial_valueConst*
valueB
 *fff?*
_output_shapes
: *
dtype0*
_class
loc:@v/dense/bias
Т
beta1_power_1
VariableV2*
_class
loc:@v/dense/bias*
dtype0*
shared_name *
shape: *
_output_shapes
: *
	container 
µ
beta1_power_1/AssignAssignbeta1_power_1beta1_power_1/initial_value*
T0*
_class
loc:@v/dense/bias*
use_locking(*
_output_shapes
: *
validate_shape(
o
beta1_power_1/readIdentitybeta1_power_1*
_class
loc:@v/dense/bias*
T0*
_output_shapes
: 
Б
beta2_power_1/initial_valueConst*
_output_shapes
: *
_class
loc:@v/dense/bias*
dtype0*
valueB
 *wЊ?
Т
beta2_power_1
VariableV2*
_class
loc:@v/dense/bias*
shared_name *
	container *
dtype0*
shape: *
_output_shapes
: 
µ
beta2_power_1/AssignAssignbeta2_power_1beta2_power_1/initial_value*
_class
loc:@v/dense/bias*
use_locking(*
_output_shapes
: *
validate_shape(*
T0
o
beta2_power_1/readIdentitybeta2_power_1*
T0*
_class
loc:@v/dense/bias*
_output_shapes
: 
Э
%v/dense/kernel/Adam/Initializer/zerosConst*
valueB@*    *
dtype0*
_output_shapes

:@*!
_class
loc:@v/dense/kernel
™
v/dense/kernel/Adam
VariableV2*
shape
:@*!
_class
loc:@v/dense/kernel*
dtype0*
_output_shapes

:@*
shared_name *
	container 
’
v/dense/kernel/Adam/AssignAssignv/dense/kernel/Adam%v/dense/kernel/Adam/Initializer/zeros*!
_class
loc:@v/dense/kernel*
use_locking(*
_output_shapes

:@*
validate_shape(*
T0
Е
v/dense/kernel/Adam/readIdentityv/dense/kernel/Adam*
T0*!
_class
loc:@v/dense/kernel*
_output_shapes

:@
Я
'v/dense/kernel/Adam_1/Initializer/zerosConst*
dtype0*
valueB@*    *!
_class
loc:@v/dense/kernel*
_output_shapes

:@
ђ
v/dense/kernel/Adam_1
VariableV2*!
_class
loc:@v/dense/kernel*
_output_shapes

:@*
dtype0*
shared_name *
shape
:@*
	container 
џ
v/dense/kernel/Adam_1/AssignAssignv/dense/kernel/Adam_1'v/dense/kernel/Adam_1/Initializer/zeros*
T0*!
_class
loc:@v/dense/kernel*
_output_shapes

:@*
use_locking(*
validate_shape(
Й
v/dense/kernel/Adam_1/readIdentityv/dense/kernel/Adam_1*!
_class
loc:@v/dense/kernel*
_output_shapes

:@*
T0
С
#v/dense/bias/Adam/Initializer/zerosConst*
valueB@*    *
_output_shapes
:@*
dtype0*
_class
loc:@v/dense/bias
Ю
v/dense/bias/Adam
VariableV2*
shape:@*
_output_shapes
:@*
_class
loc:@v/dense/bias*
shared_name *
dtype0*
	container 
…
v/dense/bias/Adam/AssignAssignv/dense/bias/Adam#v/dense/bias/Adam/Initializer/zeros*
_output_shapes
:@*
T0*
validate_shape(*
_class
loc:@v/dense/bias*
use_locking(
{
v/dense/bias/Adam/readIdentityv/dense/bias/Adam*
_output_shapes
:@*
_class
loc:@v/dense/bias*
T0
У
%v/dense/bias/Adam_1/Initializer/zerosConst*
dtype0*
_class
loc:@v/dense/bias*
_output_shapes
:@*
valueB@*    
†
v/dense/bias/Adam_1
VariableV2*
_output_shapes
:@*
shared_name *
shape:@*
dtype0*
	container *
_class
loc:@v/dense/bias
ѕ
v/dense/bias/Adam_1/AssignAssignv/dense/bias/Adam_1%v/dense/bias/Adam_1/Initializer/zeros*
T0*
validate_shape(*
_output_shapes
:@*
use_locking(*
_class
loc:@v/dense/bias

v/dense/bias/Adam_1/readIdentityv/dense/bias/Adam_1*
_output_shapes
:@*
T0*
_class
loc:@v/dense/bias
≠
7v/dense_1/kernel/Adam/Initializer/zeros/shape_as_tensorConst*#
_class
loc:@v/dense_1/kernel*
valueB"@   @   *
_output_shapes
:*
dtype0
Ч
-v/dense_1/kernel/Adam/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *#
_class
loc:@v/dense_1/kernel*
valueB
 *    
ч
'v/dense_1/kernel/Adam/Initializer/zerosFill7v/dense_1/kernel/Adam/Initializer/zeros/shape_as_tensor-v/dense_1/kernel/Adam/Initializer/zeros/Const*#
_class
loc:@v/dense_1/kernel*

index_type0*
T0*
_output_shapes

:@@
Ѓ
v/dense_1/kernel/Adam
VariableV2*
dtype0*
_output_shapes

:@@*
shape
:@@*
	container *
shared_name *#
_class
loc:@v/dense_1/kernel
Ё
v/dense_1/kernel/Adam/AssignAssignv/dense_1/kernel/Adam'v/dense_1/kernel/Adam/Initializer/zeros*
T0*
use_locking(*#
_class
loc:@v/dense_1/kernel*
_output_shapes

:@@*
validate_shape(
Л
v/dense_1/kernel/Adam/readIdentityv/dense_1/kernel/Adam*
T0*#
_class
loc:@v/dense_1/kernel*
_output_shapes

:@@
ѓ
9v/dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
valueB"@   @   *#
_class
loc:@v/dense_1/kernel*
_output_shapes
:
Щ
/v/dense_1/kernel/Adam_1/Initializer/zeros/ConstConst*
_output_shapes
: *
valueB
 *    *#
_class
loc:@v/dense_1/kernel*
dtype0
э
)v/dense_1/kernel/Adam_1/Initializer/zerosFill9v/dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensor/v/dense_1/kernel/Adam_1/Initializer/zeros/Const*

index_type0*
T0*
_output_shapes

:@@*#
_class
loc:@v/dense_1/kernel
∞
v/dense_1/kernel/Adam_1
VariableV2*
_output_shapes

:@@*
shared_name *
dtype0*#
_class
loc:@v/dense_1/kernel*
	container *
shape
:@@
г
v/dense_1/kernel/Adam_1/AssignAssignv/dense_1/kernel/Adam_1)v/dense_1/kernel/Adam_1/Initializer/zeros*
T0*#
_class
loc:@v/dense_1/kernel*
validate_shape(*
_output_shapes

:@@*
use_locking(
П
v/dense_1/kernel/Adam_1/readIdentityv/dense_1/kernel/Adam_1*
T0*
_output_shapes

:@@*#
_class
loc:@v/dense_1/kernel
Х
%v/dense_1/bias/Adam/Initializer/zerosConst*!
_class
loc:@v/dense_1/bias*
_output_shapes
:@*
dtype0*
valueB@*    
Ґ
v/dense_1/bias/Adam
VariableV2*
shape:@*
_output_shapes
:@*!
_class
loc:@v/dense_1/bias*
shared_name *
	container *
dtype0
—
v/dense_1/bias/Adam/AssignAssignv/dense_1/bias/Adam%v/dense_1/bias/Adam/Initializer/zeros*
validate_shape(*!
_class
loc:@v/dense_1/bias*
_output_shapes
:@*
T0*
use_locking(
Б
v/dense_1/bias/Adam/readIdentityv/dense_1/bias/Adam*
_output_shapes
:@*
T0*!
_class
loc:@v/dense_1/bias
Ч
'v/dense_1/bias/Adam_1/Initializer/zerosConst*
_output_shapes
:@*!
_class
loc:@v/dense_1/bias*
valueB@*    *
dtype0
§
v/dense_1/bias/Adam_1
VariableV2*
_output_shapes
:@*
shape:@*
shared_name *!
_class
loc:@v/dense_1/bias*
	container *
dtype0
„
v/dense_1/bias/Adam_1/AssignAssignv/dense_1/bias/Adam_1'v/dense_1/bias/Adam_1/Initializer/zeros*
validate_shape(*!
_class
loc:@v/dense_1/bias*
_output_shapes
:@*
use_locking(*
T0
Е
v/dense_1/bias/Adam_1/readIdentityv/dense_1/bias/Adam_1*
T0*!
_class
loc:@v/dense_1/bias*
_output_shapes
:@
°
'v/dense_2/kernel/Adam/Initializer/zerosConst*
dtype0*
valueB@*    *#
_class
loc:@v/dense_2/kernel*
_output_shapes

:@
Ѓ
v/dense_2/kernel/Adam
VariableV2*
dtype0*
shape
:@*
	container *#
_class
loc:@v/dense_2/kernel*
_output_shapes

:@*
shared_name 
Ё
v/dense_2/kernel/Adam/AssignAssignv/dense_2/kernel/Adam'v/dense_2/kernel/Adam/Initializer/zeros*
T0*
use_locking(*
validate_shape(*
_output_shapes

:@*#
_class
loc:@v/dense_2/kernel
Л
v/dense_2/kernel/Adam/readIdentityv/dense_2/kernel/Adam*
T0*#
_class
loc:@v/dense_2/kernel*
_output_shapes

:@
£
)v/dense_2/kernel/Adam_1/Initializer/zerosConst*
_output_shapes

:@*
valueB@*    *#
_class
loc:@v/dense_2/kernel*
dtype0
∞
v/dense_2/kernel/Adam_1
VariableV2*
dtype0*
	container *
_output_shapes

:@*
shared_name *#
_class
loc:@v/dense_2/kernel*
shape
:@
г
v/dense_2/kernel/Adam_1/AssignAssignv/dense_2/kernel/Adam_1)v/dense_2/kernel/Adam_1/Initializer/zeros*#
_class
loc:@v/dense_2/kernel*
validate_shape(*
use_locking(*
_output_shapes

:@*
T0
П
v/dense_2/kernel/Adam_1/readIdentityv/dense_2/kernel/Adam_1*#
_class
loc:@v/dense_2/kernel*
_output_shapes

:@*
T0
Х
%v/dense_2/bias/Adam/Initializer/zerosConst*
dtype0*
valueB*    *
_output_shapes
:*!
_class
loc:@v/dense_2/bias
Ґ
v/dense_2/bias/Adam
VariableV2*
	container *!
_class
loc:@v/dense_2/bias*
dtype0*
_output_shapes
:*
shared_name *
shape:
—
v/dense_2/bias/Adam/AssignAssignv/dense_2/bias/Adam%v/dense_2/bias/Adam/Initializer/zeros*!
_class
loc:@v/dense_2/bias*
use_locking(*
T0*
validate_shape(*
_output_shapes
:
Б
v/dense_2/bias/Adam/readIdentityv/dense_2/bias/Adam*
_output_shapes
:*
T0*!
_class
loc:@v/dense_2/bias
Ч
'v/dense_2/bias/Adam_1/Initializer/zerosConst*
valueB*    *!
_class
loc:@v/dense_2/bias*
dtype0*
_output_shapes
:
§
v/dense_2/bias/Adam_1
VariableV2*
_output_shapes
:*
	container *!
_class
loc:@v/dense_2/bias*
shape:*
shared_name *
dtype0
„
v/dense_2/bias/Adam_1/AssignAssignv/dense_2/bias/Adam_1'v/dense_2/bias/Adam_1/Initializer/zeros*!
_class
loc:@v/dense_2/bias*
use_locking(*
validate_shape(*
T0*
_output_shapes
:
Е
v/dense_2/bias/Adam_1/readIdentityv/dense_2/bias/Adam_1*
T0*
_output_shapes
:*!
_class
loc:@v/dense_2/bias
Y
Adam_1/learning_rateConst*
dtype0*
valueB
 *oГ:*
_output_shapes
: 
Q
Adam_1/beta1Const*
valueB
 *fff?*
_output_shapes
: *
dtype0
Q
Adam_1/beta2Const*
valueB
 *wЊ?*
dtype0*
_output_shapes
: 
S
Adam_1/epsilonConst*
dtype0*
_output_shapes
: *
valueB
 *wћ+2
Ў
&Adam_1/update_v/dense/kernel/ApplyAdam	ApplyAdamv/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon
Reshape_30*!
_class
loc:@v/dense/kernel*
T0*
use_nesterov( *
use_locking( *
_output_shapes

:@
 
$Adam_1/update_v/dense/bias/ApplyAdam	ApplyAdamv/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon
Reshape_31*
_class
loc:@v/dense/bias*
_output_shapes
:@*
use_locking( *
use_nesterov( *
T0
в
(Adam_1/update_v/dense_1/kernel/ApplyAdam	ApplyAdamv/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon
Reshape_32*
use_nesterov( *
T0*
_output_shapes

:@@*#
_class
loc:@v/dense_1/kernel*
use_locking( 
‘
&Adam_1/update_v/dense_1/bias/ApplyAdam	ApplyAdamv/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon
Reshape_33*
_output_shapes
:@*
use_nesterov( *!
_class
loc:@v/dense_1/bias*
use_locking( *
T0
в
(Adam_1/update_v/dense_2/kernel/ApplyAdam	ApplyAdamv/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon
Reshape_34*
_output_shapes

:@*
use_locking( *
T0*
use_nesterov( *#
_class
loc:@v/dense_2/kernel
‘
&Adam_1/update_v/dense_2/bias/ApplyAdam	ApplyAdamv/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon
Reshape_35*
use_nesterov( *
_output_shapes
:*!
_class
loc:@v/dense_2/bias*
use_locking( *
T0
н

Adam_1/mulMulbeta1_power_1/readAdam_1/beta1%^Adam_1/update_v/dense/bias/ApplyAdam'^Adam_1/update_v/dense/kernel/ApplyAdam'^Adam_1/update_v/dense_1/bias/ApplyAdam)^Adam_1/update_v/dense_1/kernel/ApplyAdam'^Adam_1/update_v/dense_2/bias/ApplyAdam)^Adam_1/update_v/dense_2/kernel/ApplyAdam*
_output_shapes
: *
_class
loc:@v/dense/bias*
T0
Э
Adam_1/AssignAssignbeta1_power_1
Adam_1/mul*
_class
loc:@v/dense/bias*
T0*
use_locking( *
_output_shapes
: *
validate_shape(
п
Adam_1/mul_1Mulbeta2_power_1/readAdam_1/beta2%^Adam_1/update_v/dense/bias/ApplyAdam'^Adam_1/update_v/dense/kernel/ApplyAdam'^Adam_1/update_v/dense_1/bias/ApplyAdam)^Adam_1/update_v/dense_1/kernel/ApplyAdam'^Adam_1/update_v/dense_2/bias/ApplyAdam)^Adam_1/update_v/dense_2/kernel/ApplyAdam*
T0*
_output_shapes
: *
_class
loc:@v/dense/bias
°
Adam_1/Assign_1Assignbeta2_power_1Adam_1/mul_1*
T0*
_class
loc:@v/dense/bias*
_output_shapes
: *
validate_shape(*
use_locking( 
®
Adam_1NoOp^Adam_1/Assign^Adam_1/Assign_1%^Adam_1/update_v/dense/bias/ApplyAdam'^Adam_1/update_v/dense/kernel/ApplyAdam'^Adam_1/update_v/dense_1/bias/ApplyAdam)^Adam_1/update_v/dense_1/kernel/ApplyAdam'^Adam_1/update_v/dense_2/bias/ApplyAdam)^Adam_1/update_v/dense_2/kernel/ApplyAdam
l
Reshape_36/shapeConst^Adam_1*
dtype0*
valueB:
€€€€€€€€€*
_output_shapes
:
p

Reshape_36Reshapev/dense/kernel/readReshape_36/shape*
Tshape0*
_output_shapes	
:А*
T0
l
Reshape_37/shapeConst^Adam_1*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€
m

Reshape_37Reshapev/dense/bias/readReshape_37/shape*
_output_shapes
:@*
Tshape0*
T0
l
Reshape_38/shapeConst^Adam_1*
dtype0*
_output_shapes
:*
valueB:
€€€€€€€€€
r

Reshape_38Reshapev/dense_1/kernel/readReshape_38/shape*
_output_shapes	
:А *
Tshape0*
T0
l
Reshape_39/shapeConst^Adam_1*
_output_shapes
:*
valueB:
€€€€€€€€€*
dtype0
o

Reshape_39Reshapev/dense_1/bias/readReshape_39/shape*
Tshape0*
_output_shapes
:@*
T0
l
Reshape_40/shapeConst^Adam_1*
dtype0*
valueB:
€€€€€€€€€*
_output_shapes
:
q

Reshape_40Reshapev/dense_2/kernel/readReshape_40/shape*
Tshape0*
T0*
_output_shapes
:@
l
Reshape_41/shapeConst^Adam_1*
_output_shapes
:*
valueB:
€€€€€€€€€*
dtype0
o

Reshape_41Reshapev/dense_2/bias/readReshape_41/shape*
Tshape0*
T0*
_output_shapes
:
X
concat_3/axisConst^Adam_1*
dtype0*
_output_shapes
: *
value	B : 
¶
concat_3ConcatV2
Reshape_36
Reshape_37
Reshape_38
Reshape_39
Reshape_40
Reshape_41concat_3/axis*
N*

Tidx0*
T0*
_output_shapes	
:Ѕ$
h
PyFunc_3PyFuncconcat_3*
Tin
2*
Tout
2*
_output_shapes
:*
token
pyfunc_3
q
Const_8Const^Adam_1*
_output_shapes
:*-
value$B""А  @      @   @      *
dtype0
\
split_3/split_dimConst^Adam_1*
dtype0*
value	B : *
_output_shapes
: 
Л
split_3SplitVPyFunc_3Const_8split_3/split_dim*
T0*

Tlen0*,
_output_shapes
::::::*
	num_split
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
Reshape_43/shapeConst^Adam_1*
valueB:@*
_output_shapes
:*
dtype0
e

Reshape_43Reshape	split_3:1Reshape_43/shape*
Tshape0*
T0*
_output_shapes
:@
j
Reshape_44/shapeConst^Adam_1*
_output_shapes
:*
valueB"@   @   *
dtype0
i

Reshape_44Reshape	split_3:2Reshape_44/shape*
Tshape0*
_output_shapes

:@@*
T0
c
Reshape_45/shapeConst^Adam_1*
dtype0*
valueB:@*
_output_shapes
:
e

Reshape_45Reshape	split_3:3Reshape_45/shape*
_output_shapes
:@*
T0*
Tshape0
j
Reshape_46/shapeConst^Adam_1*
valueB"@      *
dtype0*
_output_shapes
:
i

Reshape_46Reshape	split_3:4Reshape_46/shape*
Tshape0*
T0*
_output_shapes

:@
c
Reshape_47/shapeConst^Adam_1*
valueB:*
_output_shapes
:*
dtype0
e

Reshape_47Reshape	split_3:5Reshape_47/shape*
T0*
_output_shapes
:*
Tshape0
£
Assign_6Assignv/dense/kernel
Reshape_42*
T0*
_output_shapes

:@*
validate_shape(*
use_locking(*!
_class
loc:@v/dense/kernel
Ы
Assign_7Assignv/dense/bias
Reshape_43*
T0*
validate_shape(*
_output_shapes
:@*
use_locking(*
_class
loc:@v/dense/bias
І
Assign_8Assignv/dense_1/kernel
Reshape_44*
_output_shapes

:@@*#
_class
loc:@v/dense_1/kernel*
use_locking(*
validate_shape(*
T0
Я
Assign_9Assignv/dense_1/bias
Reshape_45*!
_class
loc:@v/dense_1/bias*
T0*
validate_shape(*
_output_shapes
:@*
use_locking(
®
	Assign_10Assignv/dense_2/kernel
Reshape_46*#
_class
loc:@v/dense_2/kernel*
use_locking(*
T0*
_output_shapes

:@*
validate_shape(
†
	Assign_11Assignv/dense_2/bias
Reshape_47*
use_locking(*
validate_shape(*
T0*
_output_shapes
:*!
_class
loc:@v/dense_2/bias
a
group_deps_2NoOp^Adam_1
^Assign_10
^Assign_11	^Assign_6	^Assign_7	^Assign_8	^Assign_9
,
group_deps_3NoOp^Adam_1^group_deps_2
т
initNoOp^beta1_power/Assign^beta1_power_1/Assign^beta2_power/Assign^beta2_power_1/Assign^pi/dense/bias/Adam/Assign^pi/dense/bias/Adam_1/Assign^pi/dense/bias/Assign^pi/dense/kernel/Adam/Assign^pi/dense/kernel/Adam_1/Assign^pi/dense/kernel/Assign^pi/dense_1/bias/Adam/Assign^pi/dense_1/bias/Adam_1/Assign^pi/dense_1/bias/Assign^pi/dense_1/kernel/Adam/Assign ^pi/dense_1/kernel/Adam_1/Assign^pi/dense_1/kernel/Assign^pi/dense_2/bias/Adam/Assign^pi/dense_2/bias/Adam_1/Assign^pi/dense_2/bias/Assign^pi/dense_2/kernel/Adam/Assign ^pi/dense_2/kernel/Adam_1/Assign^pi/dense_2/kernel/Assign^v/dense/bias/Adam/Assign^v/dense/bias/Adam_1/Assign^v/dense/bias/Assign^v/dense/kernel/Adam/Assign^v/dense/kernel/Adam_1/Assign^v/dense/kernel/Assign^v/dense_1/bias/Adam/Assign^v/dense_1/bias/Adam_1/Assign^v/dense_1/bias/Assign^v/dense_1/kernel/Adam/Assign^v/dense_1/kernel/Adam_1/Assign^v/dense_1/kernel/Assign^v/dense_2/bias/Adam/Assign^v/dense_2/bias/Adam_1/Assign^v/dense_2/bias/Assign^v/dense_2/kernel/Adam/Assign^v/dense_2/kernel/Adam_1/Assign^v/dense_2/kernel/Assign
c
Reshape_48/shapeConst*
dtype0*
_output_shapes
:*
valueB:
€€€€€€€€€
q

Reshape_48Reshapepi/dense/kernel/readReshape_48/shape*
_output_shapes	
:А*
T0*
Tshape0
c
Reshape_49/shapeConst*
valueB:
€€€€€€€€€*
_output_shapes
:*
dtype0
n

Reshape_49Reshapepi/dense/bias/readReshape_49/shape*
Tshape0*
T0*
_output_shapes
:@
c
Reshape_50/shapeConst*
valueB:
€€€€€€€€€*
_output_shapes
:*
dtype0
s

Reshape_50Reshapepi/dense_1/kernel/readReshape_50/shape*
Tshape0*
T0*
_output_shapes	
:А 
c
Reshape_51/shapeConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€
p

Reshape_51Reshapepi/dense_1/bias/readReshape_51/shape*
_output_shapes
:@*
Tshape0*
T0
c
Reshape_52/shapeConst*
valueB:
€€€€€€€€€*
_output_shapes
:*
dtype0
s

Reshape_52Reshapepi/dense_2/kernel/readReshape_52/shape*
Tshape0*
T0*
_output_shapes	
:ј
c
Reshape_53/shapeConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€
p

Reshape_53Reshapepi/dense_2/bias/readReshape_53/shape*
Tshape0*
T0*
_output_shapes
:
c
Reshape_54/shapeConst*
valueB:
€€€€€€€€€*
dtype0*
_output_shapes
:
p

Reshape_54Reshapev/dense/kernel/readReshape_54/shape*
Tshape0*
_output_shapes	
:А*
T0
c
Reshape_55/shapeConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€
m

Reshape_55Reshapev/dense/bias/readReshape_55/shape*
Tshape0*
_output_shapes
:@*
T0
c
Reshape_56/shapeConst*
dtype0*
_output_shapes
:*
valueB:
€€€€€€€€€
r

Reshape_56Reshapev/dense_1/kernel/readReshape_56/shape*
_output_shapes	
:А *
Tshape0*
T0
c
Reshape_57/shapeConst*
valueB:
€€€€€€€€€*
dtype0*
_output_shapes
:
o

Reshape_57Reshapev/dense_1/bias/readReshape_57/shape*
_output_shapes
:@*
T0*
Tshape0
c
Reshape_58/shapeConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€
q

Reshape_58Reshapev/dense_2/kernel/readReshape_58/shape*
T0*
_output_shapes
:@*
Tshape0
c
Reshape_59/shapeConst*
valueB:
€€€€€€€€€*
_output_shapes
:*
dtype0
o

Reshape_59Reshapev/dense_2/bias/readReshape_59/shape*
_output_shapes
:*
T0*
Tshape0
c
Reshape_60/shapeConst*
dtype0*
valueB:
€€€€€€€€€*
_output_shapes
:
l

Reshape_60Reshapebeta1_power/readReshape_60/shape*
_output_shapes
:*
T0*
Tshape0
c
Reshape_61/shapeConst*
dtype0*
_output_shapes
:*
valueB:
€€€€€€€€€
l

Reshape_61Reshapebeta2_power/readReshape_61/shape*
T0*
Tshape0*
_output_shapes
:
c
Reshape_62/shapeConst*
dtype0*
valueB:
€€€€€€€€€*
_output_shapes
:
v

Reshape_62Reshapepi/dense/kernel/Adam/readReshape_62/shape*
_output_shapes	
:А*
T0*
Tshape0
c
Reshape_63/shapeConst*
_output_shapes
:*
valueB:
€€€€€€€€€*
dtype0
x

Reshape_63Reshapepi/dense/kernel/Adam_1/readReshape_63/shape*
T0*
Tshape0*
_output_shapes	
:А
c
Reshape_64/shapeConst*
valueB:
€€€€€€€€€*
_output_shapes
:*
dtype0
s

Reshape_64Reshapepi/dense/bias/Adam/readReshape_64/shape*
Tshape0*
_output_shapes
:@*
T0
c
Reshape_65/shapeConst*
dtype0*
valueB:
€€€€€€€€€*
_output_shapes
:
u

Reshape_65Reshapepi/dense/bias/Adam_1/readReshape_65/shape*
Tshape0*
_output_shapes
:@*
T0
c
Reshape_66/shapeConst*
valueB:
€€€€€€€€€*
_output_shapes
:*
dtype0
x

Reshape_66Reshapepi/dense_1/kernel/Adam/readReshape_66/shape*
Tshape0*
T0*
_output_shapes	
:А 
c
Reshape_67/shapeConst*
_output_shapes
:*
valueB:
€€€€€€€€€*
dtype0
z

Reshape_67Reshapepi/dense_1/kernel/Adam_1/readReshape_67/shape*
T0*
Tshape0*
_output_shapes	
:А 
c
Reshape_68/shapeConst*
dtype0*
_output_shapes
:*
valueB:
€€€€€€€€€
u

Reshape_68Reshapepi/dense_1/bias/Adam/readReshape_68/shape*
_output_shapes
:@*
Tshape0*
T0
c
Reshape_69/shapeConst*
valueB:
€€€€€€€€€*
_output_shapes
:*
dtype0
w

Reshape_69Reshapepi/dense_1/bias/Adam_1/readReshape_69/shape*
Tshape0*
_output_shapes
:@*
T0
c
Reshape_70/shapeConst*
dtype0*
_output_shapes
:*
valueB:
€€€€€€€€€
x

Reshape_70Reshapepi/dense_2/kernel/Adam/readReshape_70/shape*
_output_shapes	
:ј*
Tshape0*
T0
c
Reshape_71/shapeConst*
valueB:
€€€€€€€€€*
_output_shapes
:*
dtype0
z

Reshape_71Reshapepi/dense_2/kernel/Adam_1/readReshape_71/shape*
Tshape0*
T0*
_output_shapes	
:ј
c
Reshape_72/shapeConst*
valueB:
€€€€€€€€€*
_output_shapes
:*
dtype0
u

Reshape_72Reshapepi/dense_2/bias/Adam/readReshape_72/shape*
Tshape0*
_output_shapes
:*
T0
c
Reshape_73/shapeConst*
dtype0*
_output_shapes
:*
valueB:
€€€€€€€€€
w

Reshape_73Reshapepi/dense_2/bias/Adam_1/readReshape_73/shape*
T0*
Tshape0*
_output_shapes
:
c
Reshape_74/shapeConst*
valueB:
€€€€€€€€€*
dtype0*
_output_shapes
:
n

Reshape_74Reshapebeta1_power_1/readReshape_74/shape*
_output_shapes
:*
T0*
Tshape0
c
Reshape_75/shapeConst*
_output_shapes
:*
valueB:
€€€€€€€€€*
dtype0
n

Reshape_75Reshapebeta2_power_1/readReshape_75/shape*
T0*
Tshape0*
_output_shapes
:
c
Reshape_76/shapeConst*
dtype0*
valueB:
€€€€€€€€€*
_output_shapes
:
u

Reshape_76Reshapev/dense/kernel/Adam/readReshape_76/shape*
Tshape0*
T0*
_output_shapes	
:А
c
Reshape_77/shapeConst*
valueB:
€€€€€€€€€*
dtype0*
_output_shapes
:
w

Reshape_77Reshapev/dense/kernel/Adam_1/readReshape_77/shape*
Tshape0*
T0*
_output_shapes	
:А
c
Reshape_78/shapeConst*
dtype0*
_output_shapes
:*
valueB:
€€€€€€€€€
r

Reshape_78Reshapev/dense/bias/Adam/readReshape_78/shape*
Tshape0*
T0*
_output_shapes
:@
c
Reshape_79/shapeConst*
valueB:
€€€€€€€€€*
dtype0*
_output_shapes
:
t

Reshape_79Reshapev/dense/bias/Adam_1/readReshape_79/shape*
T0*
_output_shapes
:@*
Tshape0
c
Reshape_80/shapeConst*
dtype0*
valueB:
€€€€€€€€€*
_output_shapes
:
w

Reshape_80Reshapev/dense_1/kernel/Adam/readReshape_80/shape*
Tshape0*
_output_shapes	
:А *
T0
c
Reshape_81/shapeConst*
valueB:
€€€€€€€€€*
_output_shapes
:*
dtype0
y

Reshape_81Reshapev/dense_1/kernel/Adam_1/readReshape_81/shape*
T0*
_output_shapes	
:А *
Tshape0
c
Reshape_82/shapeConst*
_output_shapes
:*
valueB:
€€€€€€€€€*
dtype0
t

Reshape_82Reshapev/dense_1/bias/Adam/readReshape_82/shape*
Tshape0*
_output_shapes
:@*
T0
c
Reshape_83/shapeConst*
dtype0*
valueB:
€€€€€€€€€*
_output_shapes
:
v

Reshape_83Reshapev/dense_1/bias/Adam_1/readReshape_83/shape*
T0*
_output_shapes
:@*
Tshape0
c
Reshape_84/shapeConst*
dtype0*
valueB:
€€€€€€€€€*
_output_shapes
:
v

Reshape_84Reshapev/dense_2/kernel/Adam/readReshape_84/shape*
Tshape0*
T0*
_output_shapes
:@
c
Reshape_85/shapeConst*
valueB:
€€€€€€€€€*
_output_shapes
:*
dtype0
x

Reshape_85Reshapev/dense_2/kernel/Adam_1/readReshape_85/shape*
T0*
Tshape0*
_output_shapes
:@
c
Reshape_86/shapeConst*
_output_shapes
:*
valueB:
€€€€€€€€€*
dtype0
t

Reshape_86Reshapev/dense_2/bias/Adam/readReshape_86/shape*
Tshape0*
_output_shapes
:*
T0
c
Reshape_87/shapeConst*
dtype0*
_output_shapes
:*
valueB:
€€€€€€€€€
v

Reshape_87Reshapev/dense_2/bias/Adam_1/readReshape_87/shape*
Tshape0*
T0*
_output_shapes
:
O
concat_4/axisConst*
_output_shapes
: *
dtype0*
value	B : 
њ
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
Reshape_87concat_4/axis*

Tidx0*
T0*
N(*
_output_shapes

:Рё
h
PyFunc_4PyFuncconcat_4*
Tin
2*
Tout
2*
token
pyfunc_4*
_output_shapes
:
ф
Const_9Const*Є
valueЃBЂ("†А  @      @   ј      А  @      @   @            А  А  @   @         @   @   ј   ј               А  А  @   @         @   @   @   @         *
_output_shapes
:(*
dtype0
S
split_4/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
Ц
split_4SplitVPyFunc_4Const_9split_4/split_dim*
	num_split(*

Tlen0*ґ
_output_shapes£
†::::::::::::::::::::::::::::::::::::::::*
T0
a
Reshape_88/shapeConst*
dtype0*
valueB"   @   *
_output_shapes
:
g

Reshape_88Reshapesplit_4Reshape_88/shape*
T0*
_output_shapes

:@*
Tshape0
Z
Reshape_89/shapeConst*
dtype0*
valueB:@*
_output_shapes
:
e

Reshape_89Reshape	split_4:1Reshape_89/shape*
Tshape0*
_output_shapes
:@*
T0
a
Reshape_90/shapeConst*
dtype0*
valueB"@   @   *
_output_shapes
:
i

Reshape_90Reshape	split_4:2Reshape_90/shape*
T0*
_output_shapes

:@@*
Tshape0
Z
Reshape_91/shapeConst*
valueB:@*
dtype0*
_output_shapes
:
e

Reshape_91Reshape	split_4:3Reshape_91/shape*
T0*
Tshape0*
_output_shapes
:@
a
Reshape_92/shapeConst*
valueB"@      *
_output_shapes
:*
dtype0
i

Reshape_92Reshape	split_4:4Reshape_92/shape*
T0*
_output_shapes

:@*
Tshape0
Z
Reshape_93/shapeConst*
valueB:*
_output_shapes
:*
dtype0
e

Reshape_93Reshape	split_4:5Reshape_93/shape*
Tshape0*
T0*
_output_shapes
:
a
Reshape_94/shapeConst*
valueB"   @   *
_output_shapes
:*
dtype0
i

Reshape_94Reshape	split_4:6Reshape_94/shape*
T0*
Tshape0*
_output_shapes

:@
Z
Reshape_95/shapeConst*
valueB:@*
dtype0*
_output_shapes
:
e

Reshape_95Reshape	split_4:7Reshape_95/shape*
T0*
Tshape0*
_output_shapes
:@
a
Reshape_96/shapeConst*
dtype0*
_output_shapes
:*
valueB"@   @   
i

Reshape_96Reshape	split_4:8Reshape_96/shape*
_output_shapes

:@@*
T0*
Tshape0
Z
Reshape_97/shapeConst*
dtype0*
_output_shapes
:*
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

:@*
T0*
Tshape0
Z
Reshape_99/shapeConst*
dtype0*
_output_shapes
:*
valueB:
f

Reshape_99Reshape
split_4:11Reshape_99/shape*
_output_shapes
:*
Tshape0*
T0
T
Reshape_100/shapeConst*
dtype0*
valueB *
_output_shapes
: 
d
Reshape_100Reshape
split_4:12Reshape_100/shape*
T0*
_output_shapes
: *
Tshape0
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
valueB"   @   *
dtype0
l
Reshape_102Reshape
split_4:14Reshape_102/shape*
Tshape0*
_output_shapes

:@*
T0
b
Reshape_103/shapeConst*
valueB"   @   *
_output_shapes
:*
dtype0
l
Reshape_103Reshape
split_4:15Reshape_103/shape*
T0*
Tshape0*
_output_shapes

:@
[
Reshape_104/shapeConst*
valueB:@*
dtype0*
_output_shapes
:
h
Reshape_104Reshape
split_4:16Reshape_104/shape*
Tshape0*
_output_shapes
:@*
T0
[
Reshape_105/shapeConst*
_output_shapes
:*
valueB:@*
dtype0
h
Reshape_105Reshape
split_4:17Reshape_105/shape*
T0*
_output_shapes
:@*
Tshape0
b
Reshape_106/shapeConst*
dtype0*
_output_shapes
:*
valueB"@   @   
l
Reshape_106Reshape
split_4:18Reshape_106/shape*
T0*
Tshape0*
_output_shapes

:@@
b
Reshape_107/shapeConst*
dtype0*
valueB"@   @   *
_output_shapes
:
l
Reshape_107Reshape
split_4:19Reshape_107/shape*
T0*
_output_shapes

:@@*
Tshape0
[
Reshape_108/shapeConst*
_output_shapes
:*
dtype0*
valueB:@
h
Reshape_108Reshape
split_4:20Reshape_108/shape*
Tshape0*
T0*
_output_shapes
:@
[
Reshape_109/shapeConst*
valueB:@*
dtype0*
_output_shapes
:
h
Reshape_109Reshape
split_4:21Reshape_109/shape*
_output_shapes
:@*
Tshape0*
T0
b
Reshape_110/shapeConst*
_output_shapes
:*
dtype0*
valueB"@      
l
Reshape_110Reshape
split_4:22Reshape_110/shape*
_output_shapes

:@*
T0*
Tshape0
b
Reshape_111/shapeConst*
_output_shapes
:*
valueB"@      *
dtype0
l
Reshape_111Reshape
split_4:23Reshape_111/shape*
_output_shapes

:@*
T0*
Tshape0
[
Reshape_112/shapeConst*
valueB:*
_output_shapes
:*
dtype0
h
Reshape_112Reshape
split_4:24Reshape_112/shape*
_output_shapes
:*
T0*
Tshape0
[
Reshape_113/shapeConst*
_output_shapes
:*
valueB:*
dtype0
h
Reshape_113Reshape
split_4:25Reshape_113/shape*
T0*
_output_shapes
:*
Tshape0
T
Reshape_114/shapeConst*
dtype0*
_output_shapes
: *
valueB 
d
Reshape_114Reshape
split_4:26Reshape_114/shape*
T0*
_output_shapes
: *
Tshape0
T
Reshape_115/shapeConst*
valueB *
_output_shapes
: *
dtype0
d
Reshape_115Reshape
split_4:27Reshape_115/shape*
T0*
_output_shapes
: *
Tshape0
b
Reshape_116/shapeConst*
dtype0*
_output_shapes
:*
valueB"   @   
l
Reshape_116Reshape
split_4:28Reshape_116/shape*
T0*
_output_shapes

:@*
Tshape0
b
Reshape_117/shapeConst*
dtype0*
_output_shapes
:*
valueB"   @   
l
Reshape_117Reshape
split_4:29Reshape_117/shape*
Tshape0*
T0*
_output_shapes

:@
[
Reshape_118/shapeConst*
_output_shapes
:*
dtype0*
valueB:@
h
Reshape_118Reshape
split_4:30Reshape_118/shape*
_output_shapes
:@*
T0*
Tshape0
[
Reshape_119/shapeConst*
_output_shapes
:*
dtype0*
valueB:@
h
Reshape_119Reshape
split_4:31Reshape_119/shape*
_output_shapes
:@*
T0*
Tshape0
b
Reshape_120/shapeConst*
_output_shapes
:*
dtype0*
valueB"@   @   
l
Reshape_120Reshape
split_4:32Reshape_120/shape*
_output_shapes

:@@*
T0*
Tshape0
b
Reshape_121/shapeConst*
dtype0*
valueB"@   @   *
_output_shapes
:
l
Reshape_121Reshape
split_4:33Reshape_121/shape*
Tshape0*
_output_shapes

:@@*
T0
[
Reshape_122/shapeConst*
valueB:@*
_output_shapes
:*
dtype0
h
Reshape_122Reshape
split_4:34Reshape_122/shape*
_output_shapes
:@*
T0*
Tshape0
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
:@*
T0*
Tshape0
b
Reshape_124/shapeConst*
_output_shapes
:*
valueB"@      *
dtype0
l
Reshape_124Reshape
split_4:36Reshape_124/shape*
T0*
_output_shapes

:@*
Tshape0
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
split_4:38Reshape_126/shape*
_output_shapes
:*
Tshape0*
T0
[
Reshape_127/shapeConst*
_output_shapes
:*
valueB:*
dtype0
h
Reshape_127Reshape
split_4:39Reshape_127/shape*
_output_shapes
:*
T0*
Tshape0
¶
	Assign_12Assignpi/dense/kernel
Reshape_88*
_output_shapes

:@*
validate_shape(*
use_locking(*"
_class
loc:@pi/dense/kernel*
T0
Ю
	Assign_13Assignpi/dense/bias
Reshape_89*
use_locking(* 
_class
loc:@pi/dense/bias*
T0*
_output_shapes
:@*
validate_shape(
™
	Assign_14Assignpi/dense_1/kernel
Reshape_90*
T0*
use_locking(*
validate_shape(*
_output_shapes

:@@*$
_class
loc:@pi/dense_1/kernel
Ґ
	Assign_15Assignpi/dense_1/bias
Reshape_91*
_output_shapes
:@*
T0*"
_class
loc:@pi/dense_1/bias*
use_locking(*
validate_shape(
™
	Assign_16Assignpi/dense_2/kernel
Reshape_92*
use_locking(*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@*
T0
Ґ
	Assign_17Assignpi/dense_2/bias
Reshape_93*"
_class
loc:@pi/dense_2/bias*
T0*
validate_shape(*
use_locking(*
_output_shapes
:
§
	Assign_18Assignv/dense/kernel
Reshape_94*!
_class
loc:@v/dense/kernel*
T0*
use_locking(*
_output_shapes

:@*
validate_shape(
Ь
	Assign_19Assignv/dense/bias
Reshape_95*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*
_class
loc:@v/dense/bias
®
	Assign_20Assignv/dense_1/kernel
Reshape_96*
_output_shapes

:@@*#
_class
loc:@v/dense_1/kernel*
use_locking(*
validate_shape(*
T0
†
	Assign_21Assignv/dense_1/bias
Reshape_97*
validate_shape(*
T0*
use_locking(*
_output_shapes
:@*!
_class
loc:@v/dense_1/bias
®
	Assign_22Assignv/dense_2/kernel
Reshape_98*
_output_shapes

:@*
T0*#
_class
loc:@v/dense_2/kernel*
use_locking(*
validate_shape(
†
	Assign_23Assignv/dense_2/bias
Reshape_99*
use_locking(*!
_class
loc:@v/dense_2/bias*
_output_shapes
:*
validate_shape(*
T0
Щ
	Assign_24Assignbeta1_powerReshape_100*
use_locking(*
T0*
_output_shapes
: *
validate_shape(* 
_class
loc:@pi/dense/bias
Щ
	Assign_25Assignbeta2_powerReshape_101*
_output_shapes
: *
use_locking(* 
_class
loc:@pi/dense/bias*
T0*
validate_shape(
ђ
	Assign_26Assignpi/dense/kernel/AdamReshape_102*
validate_shape(*
use_locking(*"
_class
loc:@pi/dense/kernel*
_output_shapes

:@*
T0
Ѓ
	Assign_27Assignpi/dense/kernel/Adam_1Reshape_103*
validate_shape(*
use_locking(*
T0*
_output_shapes

:@*"
_class
loc:@pi/dense/kernel
§
	Assign_28Assignpi/dense/bias/AdamReshape_104*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
_output_shapes
:@*
validate_shape(
¶
	Assign_29Assignpi/dense/bias/Adam_1Reshape_105*
_output_shapes
:@*
use_locking(*
validate_shape(* 
_class
loc:@pi/dense/bias*
T0
∞
	Assign_30Assignpi/dense_1/kernel/AdamReshape_106*
T0*
_output_shapes

:@@*$
_class
loc:@pi/dense_1/kernel*
use_locking(*
validate_shape(
≤
	Assign_31Assignpi/dense_1/kernel/Adam_1Reshape_107*
T0*
use_locking(*
validate_shape(*
_output_shapes

:@@*$
_class
loc:@pi/dense_1/kernel
®
	Assign_32Assignpi/dense_1/bias/AdamReshape_108*
T0*
use_locking(*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
_output_shapes
:@
™
	Assign_33Assignpi/dense_1/bias/Adam_1Reshape_109*
validate_shape(*"
_class
loc:@pi/dense_1/bias*
T0*
_output_shapes
:@*
use_locking(
∞
	Assign_34Assignpi/dense_2/kernel/AdamReshape_110*
use_locking(*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
T0*
_output_shapes

:@
≤
	Assign_35Assignpi/dense_2/kernel/Adam_1Reshape_111*
_output_shapes

:@*
validate_shape(*
T0*$
_class
loc:@pi/dense_2/kernel*
use_locking(
®
	Assign_36Assignpi/dense_2/bias/AdamReshape_112*
validate_shape(*
T0*
use_locking(*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:
™
	Assign_37Assignpi/dense_2/bias/Adam_1Reshape_113*"
_class
loc:@pi/dense_2/bias*
T0*
use_locking(*
_output_shapes
:*
validate_shape(
Ъ
	Assign_38Assignbeta1_power_1Reshape_114*
_output_shapes
: *
_class
loc:@v/dense/bias*
T0*
validate_shape(*
use_locking(
Ъ
	Assign_39Assignbeta2_power_1Reshape_115*
use_locking(*
_class
loc:@v/dense/bias*
_output_shapes
: *
T0*
validate_shape(
™
	Assign_40Assignv/dense/kernel/AdamReshape_116*
_output_shapes

:@*!
_class
loc:@v/dense/kernel*
T0*
use_locking(*
validate_shape(
ђ
	Assign_41Assignv/dense/kernel/Adam_1Reshape_117*
use_locking(*
_output_shapes

:@*
validate_shape(*
T0*!
_class
loc:@v/dense/kernel
Ґ
	Assign_42Assignv/dense/bias/AdamReshape_118*
T0*
use_locking(*
_output_shapes
:@*
validate_shape(*
_class
loc:@v/dense/bias
§
	Assign_43Assignv/dense/bias/Adam_1Reshape_119*
_output_shapes
:@*
_class
loc:@v/dense/bias*
T0*
use_locking(*
validate_shape(
Ѓ
	Assign_44Assignv/dense_1/kernel/AdamReshape_120*#
_class
loc:@v/dense_1/kernel*
validate_shape(*
use_locking(*
_output_shapes

:@@*
T0
∞
	Assign_45Assignv/dense_1/kernel/Adam_1Reshape_121*#
_class
loc:@v/dense_1/kernel*
T0*
use_locking(*
validate_shape(*
_output_shapes

:@@
¶
	Assign_46Assignv/dense_1/bias/AdamReshape_122*
T0*!
_class
loc:@v/dense_1/bias*
validate_shape(*
_output_shapes
:@*
use_locking(
®
	Assign_47Assignv/dense_1/bias/Adam_1Reshape_123*
validate_shape(*
_output_shapes
:@*
use_locking(*!
_class
loc:@v/dense_1/bias*
T0
Ѓ
	Assign_48Assignv/dense_2/kernel/AdamReshape_124*
_output_shapes

:@*
T0*
use_locking(*#
_class
loc:@v/dense_2/kernel*
validate_shape(
∞
	Assign_49Assignv/dense_2/kernel/Adam_1Reshape_125*
_output_shapes

:@*
use_locking(*#
_class
loc:@v/dense_2/kernel*
T0*
validate_shape(
¶
	Assign_50Assignv/dense_2/bias/AdamReshape_126*
validate_shape(*!
_class
loc:@v/dense_2/bias*
use_locking(*
T0*
_output_shapes
:
®
	Assign_51Assignv/dense_2/bias/Adam_1Reshape_127*
T0*
validate_shape(*!
_class
loc:@v/dense_2/bias*
use_locking(*
_output_shapes
:
ф
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
save/filename/inputConst*
dtype0*
valueB Bmodel*
_output_shapes
: 
n
save/filenamePlaceholderWithDefaultsave/filename/input*
shape: *
dtype0*
_output_shapes
: 
e

save/ConstPlaceholderWithDefaultsave/filename*
_output_shapes
: *
shape: *
dtype0
Д
save/StringJoin/inputs_1Const*
_output_shapes
: *<
value3B1 B+_temp_a42bf57eaaac4472a3b5183d45bc89f2/part*
dtype0
u
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
_output_shapes
: *
	separator *
N
Q
save/num_shardsConst*
value	B :*
_output_shapes
: *
dtype0
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
И
save/SaveV2/tensor_namesConst*
dtype0*ї
value±BЃ(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
_output_shapes
:(
≥
save/SaveV2/shape_and_slicesConst*
_output_shapes
:(*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
Ѕ
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1v/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1v/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1v/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1v/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1v/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1v/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1*6
dtypes,
*2(
С
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2*'
_class
loc:@save/ShardedFilename*
T0*
_output_shapes
: 
Э
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency*
_output_shapes
:*
N*

axis *
T0
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
Л
save/RestoreV2/tensor_namesConst*
_output_shapes
:(*ї
value±BЃ(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
dtype0
ґ
save/RestoreV2/shape_and_slicesConst*
_output_shapes
:(*
dtype0*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
÷
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*6
dtypes,
*2(*ґ
_output_shapes£
†::::::::::::::::::::::::::::::::::::::::
Ю
save/AssignAssignbeta1_powersave/RestoreV2* 
_class
loc:@pi/dense/bias*
T0*
validate_shape(*
use_locking(*
_output_shapes
: 
£
save/Assign_1Assignbeta1_power_1save/RestoreV2:1*
_class
loc:@v/dense/bias*
validate_shape(*
T0*
use_locking(*
_output_shapes
: 
Ґ
save/Assign_2Assignbeta2_powersave/RestoreV2:2*
T0*
_output_shapes
: * 
_class
loc:@pi/dense/bias*
use_locking(*
validate_shape(
£
save/Assign_3Assignbeta2_power_1save/RestoreV2:3*
T0*
validate_shape(*
_output_shapes
: *
use_locking(*
_class
loc:@v/dense/bias
®
save/Assign_4Assignpi/dense/biassave/RestoreV2:4*
validate_shape(*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
_output_shapes
:@
≠
save/Assign_5Assignpi/dense/bias/Adamsave/RestoreV2:5*
T0* 
_class
loc:@pi/dense/bias*
use_locking(*
_output_shapes
:@*
validate_shape(
ѓ
save/Assign_6Assignpi/dense/bias/Adam_1save/RestoreV2:6*
T0*
use_locking(*
_output_shapes
:@* 
_class
loc:@pi/dense/bias*
validate_shape(
∞
save/Assign_7Assignpi/dense/kernelsave/RestoreV2:7*
use_locking(*
validate_shape(*"
_class
loc:@pi/dense/kernel*
_output_shapes

:@*
T0
µ
save/Assign_8Assignpi/dense/kernel/Adamsave/RestoreV2:8*
use_locking(*
T0*
_output_shapes

:@*"
_class
loc:@pi/dense/kernel*
validate_shape(
Ј
save/Assign_9Assignpi/dense/kernel/Adam_1save/RestoreV2:9*
validate_shape(*
use_locking(*"
_class
loc:@pi/dense/kernel*
_output_shapes

:@*
T0
Ѓ
save/Assign_10Assignpi/dense_1/biassave/RestoreV2:10*
use_locking(*
validate_shape(*"
_class
loc:@pi/dense_1/bias*
T0*
_output_shapes
:@
≥
save/Assign_11Assignpi/dense_1/bias/Adamsave/RestoreV2:11*
T0*
_output_shapes
:@*
use_locking(*"
_class
loc:@pi/dense_1/bias*
validate_shape(
µ
save/Assign_12Assignpi/dense_1/bias/Adam_1save/RestoreV2:12*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0
ґ
save/Assign_13Assignpi/dense_1/kernelsave/RestoreV2:13*
validate_shape(*
use_locking(*
_output_shapes

:@@*
T0*$
_class
loc:@pi/dense_1/kernel
ї
save/Assign_14Assignpi/dense_1/kernel/Adamsave/RestoreV2:14*
use_locking(*
T0*
validate_shape(*
_output_shapes

:@@*$
_class
loc:@pi/dense_1/kernel
љ
save/Assign_15Assignpi/dense_1/kernel/Adam_1save/RestoreV2:15*$
_class
loc:@pi/dense_1/kernel*
_output_shapes

:@@*
use_locking(*
T0*
validate_shape(
Ѓ
save/Assign_16Assignpi/dense_2/biassave/RestoreV2:16*
validate_shape(*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:*
T0*
use_locking(
≥
save/Assign_17Assignpi/dense_2/bias/Adamsave/RestoreV2:17*
_output_shapes
:*
use_locking(*"
_class
loc:@pi/dense_2/bias*
T0*
validate_shape(
µ
save/Assign_18Assignpi/dense_2/bias/Adam_1save/RestoreV2:18*
_output_shapes
:*
validate_shape(*
use_locking(*"
_class
loc:@pi/dense_2/bias*
T0
ґ
save/Assign_19Assignpi/dense_2/kernelsave/RestoreV2:19*
T0*
validate_shape(*
use_locking(*
_output_shapes

:@*$
_class
loc:@pi/dense_2/kernel
ї
save/Assign_20Assignpi/dense_2/kernel/Adamsave/RestoreV2:20*
_output_shapes

:@*$
_class
loc:@pi/dense_2/kernel*
use_locking(*
T0*
validate_shape(
љ
save/Assign_21Assignpi/dense_2/kernel/Adam_1save/RestoreV2:21*
T0*
_output_shapes

:@*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
use_locking(
®
save/Assign_22Assignv/dense/biassave/RestoreV2:22*
T0*
_output_shapes
:@*
validate_shape(*
use_locking(*
_class
loc:@v/dense/bias
≠
save/Assign_23Assignv/dense/bias/Adamsave/RestoreV2:23*
validate_shape(*
_class
loc:@v/dense/bias*
_output_shapes
:@*
use_locking(*
T0
ѓ
save/Assign_24Assignv/dense/bias/Adam_1save/RestoreV2:24*
_class
loc:@v/dense/bias*
use_locking(*
_output_shapes
:@*
T0*
validate_shape(
∞
save/Assign_25Assignv/dense/kernelsave/RestoreV2:25*
use_locking(*!
_class
loc:@v/dense/kernel*
validate_shape(*
T0*
_output_shapes

:@
µ
save/Assign_26Assignv/dense/kernel/Adamsave/RestoreV2:26*
validate_shape(*!
_class
loc:@v/dense/kernel*
_output_shapes

:@*
T0*
use_locking(
Ј
save/Assign_27Assignv/dense/kernel/Adam_1save/RestoreV2:27*
_output_shapes

:@*
validate_shape(*
use_locking(*!
_class
loc:@v/dense/kernel*
T0
ђ
save/Assign_28Assignv/dense_1/biassave/RestoreV2:28*
use_locking(*
_output_shapes
:@*!
_class
loc:@v/dense_1/bias*
validate_shape(*
T0
±
save/Assign_29Assignv/dense_1/bias/Adamsave/RestoreV2:29*
use_locking(*
_output_shapes
:@*
T0*!
_class
loc:@v/dense_1/bias*
validate_shape(
≥
save/Assign_30Assignv/dense_1/bias/Adam_1save/RestoreV2:30*
validate_shape(*
_output_shapes
:@*
T0*!
_class
loc:@v/dense_1/bias*
use_locking(
і
save/Assign_31Assignv/dense_1/kernelsave/RestoreV2:31*
validate_shape(*
_output_shapes

:@@*
T0*
use_locking(*#
_class
loc:@v/dense_1/kernel
є
save/Assign_32Assignv/dense_1/kernel/Adamsave/RestoreV2:32*#
_class
loc:@v/dense_1/kernel*
validate_shape(*
T0*
use_locking(*
_output_shapes

:@@
ї
save/Assign_33Assignv/dense_1/kernel/Adam_1save/RestoreV2:33*
use_locking(*
_output_shapes

:@@*
T0*#
_class
loc:@v/dense_1/kernel*
validate_shape(
ђ
save/Assign_34Assignv/dense_2/biassave/RestoreV2:34*!
_class
loc:@v/dense_2/bias*
use_locking(*
_output_shapes
:*
validate_shape(*
T0
±
save/Assign_35Assignv/dense_2/bias/Adamsave/RestoreV2:35*!
_class
loc:@v/dense_2/bias*
use_locking(*
validate_shape(*
T0*
_output_shapes
:
≥
save/Assign_36Assignv/dense_2/bias/Adam_1save/RestoreV2:36*!
_class
loc:@v/dense_2/bias*
validate_shape(*
use_locking(*
_output_shapes
:*
T0
і
save/Assign_37Assignv/dense_2/kernelsave/RestoreV2:37*
validate_shape(*
use_locking(*
_output_shapes

:@*
T0*#
_class
loc:@v/dense_2/kernel
є
save/Assign_38Assignv/dense_2/kernel/Adamsave/RestoreV2:38*
validate_shape(*#
_class
loc:@v/dense_2/kernel*
use_locking(*
_output_shapes

:@*
T0
ї
save/Assign_39Assignv/dense_2/kernel/Adam_1save/RestoreV2:39*
_output_shapes

:@*
use_locking(*
validate_shape(*#
_class
loc:@v/dense_2/kernel*
T0
ґ
save/restore_shardNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_2^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25^save/Assign_26^save/Assign_27^save/Assign_28^save/Assign_29^save/Assign_3^save/Assign_30^save/Assign_31^save/Assign_32^save/Assign_33^save/Assign_34^save/Assign_35^save/Assign_36^save/Assign_37^save/Assign_38^save/Assign_39^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9
-
save/restore_allNoOp^save/restore_shard
[
save_1/filename/inputConst*
_output_shapes
: *
dtype0*
valueB Bmodel
r
save_1/filenamePlaceholderWithDefaultsave_1/filename/input*
_output_shapes
: *
dtype0*
shape: 
i
save_1/ConstPlaceholderWithDefaultsave_1/filename*
_output_shapes
: *
dtype0*
shape: 
Ж
save_1/StringJoin/inputs_1Const*
dtype0*
_output_shapes
: *<
value3B1 B+_temp_a95b6524d1a8453c9c039b3a15e6fd72/part
{
save_1/StringJoin
StringJoinsave_1/Constsave_1/StringJoin/inputs_1*
	separator *
_output_shapes
: *
N
S
save_1/num_shardsConst*
value	B :*
_output_shapes
: *
dtype0
^
save_1/ShardedFilename/shardConst*
dtype0*
value	B : *
_output_shapes
: 
Е
save_1/ShardedFilenameShardedFilenamesave_1/StringJoinsave_1/ShardedFilename/shardsave_1/num_shards*
_output_shapes
: 
К
save_1/SaveV2/tensor_namesConst*ї
value±BЃ(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
_output_shapes
:(*
dtype0
µ
save_1/SaveV2/shape_and_slicesConst*
dtype0*
_output_shapes
:(*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
…
save_1/SaveV2SaveV2save_1/ShardedFilenamesave_1/SaveV2/tensor_namessave_1/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1v/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1v/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1v/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1v/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1v/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1v/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1*6
dtypes,
*2(
Щ
save_1/control_dependencyIdentitysave_1/ShardedFilename^save_1/SaveV2*
T0*)
_class
loc:@save_1/ShardedFilename*
_output_shapes
: 
£
-save_1/MergeV2Checkpoints/checkpoint_prefixesPacksave_1/ShardedFilename^save_1/control_dependency*

axis *
_output_shapes
:*
N*
T0
Г
save_1/MergeV2CheckpointsMergeV2Checkpoints-save_1/MergeV2Checkpoints/checkpoint_prefixessave_1/Const*
delete_old_dirs(
В
save_1/IdentityIdentitysave_1/Const^save_1/MergeV2Checkpoints^save_1/control_dependency*
_output_shapes
: *
T0
Н
save_1/RestoreV2/tensor_namesConst*ї
value±BЃ(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:(
Є
!save_1/RestoreV2/shape_and_slicesConst*
_output_shapes
:(*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
ё
save_1/RestoreV2	RestoreV2save_1/Constsave_1/RestoreV2/tensor_names!save_1/RestoreV2/shape_and_slices*6
dtypes,
*2(*ґ
_output_shapes£
†::::::::::::::::::::::::::::::::::::::::
Ґ
save_1/AssignAssignbeta1_powersave_1/RestoreV2* 
_class
loc:@pi/dense/bias*
T0*
validate_shape(*
_output_shapes
: *
use_locking(
І
save_1/Assign_1Assignbeta1_power_1save_1/RestoreV2:1*
validate_shape(*
T0*
_class
loc:@v/dense/bias*
_output_shapes
: *
use_locking(
¶
save_1/Assign_2Assignbeta2_powersave_1/RestoreV2:2* 
_class
loc:@pi/dense/bias*
validate_shape(*
T0*
_output_shapes
: *
use_locking(
І
save_1/Assign_3Assignbeta2_power_1save_1/RestoreV2:3*
use_locking(*
T0*
_output_shapes
: *
validate_shape(*
_class
loc:@v/dense/bias
ђ
save_1/Assign_4Assignpi/dense/biassave_1/RestoreV2:4* 
_class
loc:@pi/dense/bias*
T0*
validate_shape(*
_output_shapes
:@*
use_locking(
±
save_1/Assign_5Assignpi/dense/bias/Adamsave_1/RestoreV2:5*
use_locking(*
_output_shapes
:@*
validate_shape(* 
_class
loc:@pi/dense/bias*
T0
≥
save_1/Assign_6Assignpi/dense/bias/Adam_1save_1/RestoreV2:6*
validate_shape(*
T0*
_output_shapes
:@*
use_locking(* 
_class
loc:@pi/dense/bias
і
save_1/Assign_7Assignpi/dense/kernelsave_1/RestoreV2:7*
_output_shapes

:@*
T0*
validate_shape(*"
_class
loc:@pi/dense/kernel*
use_locking(
є
save_1/Assign_8Assignpi/dense/kernel/Adamsave_1/RestoreV2:8*
validate_shape(*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel*
_output_shapes

:@
ї
save_1/Assign_9Assignpi/dense/kernel/Adam_1save_1/RestoreV2:9*
validate_shape(*"
_class
loc:@pi/dense/kernel*
T0*
use_locking(*
_output_shapes

:@
≤
save_1/Assign_10Assignpi/dense_1/biassave_1/RestoreV2:10*
_output_shapes
:@*
T0*"
_class
loc:@pi/dense_1/bias*
use_locking(*
validate_shape(
Ј
save_1/Assign_11Assignpi/dense_1/bias/Adamsave_1/RestoreV2:11*"
_class
loc:@pi/dense_1/bias*
T0*
validate_shape(*
use_locking(*
_output_shapes
:@
є
save_1/Assign_12Assignpi/dense_1/bias/Adam_1save_1/RestoreV2:12*"
_class
loc:@pi/dense_1/bias*
T0*
use_locking(*
_output_shapes
:@*
validate_shape(
Ї
save_1/Assign_13Assignpi/dense_1/kernelsave_1/RestoreV2:13*$
_class
loc:@pi/dense_1/kernel*
validate_shape(*
use_locking(*
T0*
_output_shapes

:@@
њ
save_1/Assign_14Assignpi/dense_1/kernel/Adamsave_1/RestoreV2:14*
_output_shapes

:@@*
validate_shape(*
T0*
use_locking(*$
_class
loc:@pi/dense_1/kernel
Ѕ
save_1/Assign_15Assignpi/dense_1/kernel/Adam_1save_1/RestoreV2:15*
use_locking(*
validate_shape(*
T0*
_output_shapes

:@@*$
_class
loc:@pi/dense_1/kernel
≤
save_1/Assign_16Assignpi/dense_2/biassave_1/RestoreV2:16*
use_locking(*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:*
T0*
validate_shape(
Ј
save_1/Assign_17Assignpi/dense_2/bias/Adamsave_1/RestoreV2:17*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:*
use_locking(*
validate_shape(*
T0
є
save_1/Assign_18Assignpi/dense_2/bias/Adam_1save_1/RestoreV2:18*
validate_shape(*
use_locking(*"
_class
loc:@pi/dense_2/bias*
T0*
_output_shapes
:
Ї
save_1/Assign_19Assignpi/dense_2/kernelsave_1/RestoreV2:19*
validate_shape(*$
_class
loc:@pi/dense_2/kernel*
T0*
_output_shapes

:@*
use_locking(
њ
save_1/Assign_20Assignpi/dense_2/kernel/Adamsave_1/RestoreV2:20*
use_locking(*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@*
T0
Ѕ
save_1/Assign_21Assignpi/dense_2/kernel/Adam_1save_1/RestoreV2:21*$
_class
loc:@pi/dense_2/kernel*
use_locking(*
_output_shapes

:@*
T0*
validate_shape(
ђ
save_1/Assign_22Assignv/dense/biassave_1/RestoreV2:22*
_output_shapes
:@*
_class
loc:@v/dense/bias*
validate_shape(*
use_locking(*
T0
±
save_1/Assign_23Assignv/dense/bias/Adamsave_1/RestoreV2:23*
use_locking(*
T0*
validate_shape(*
_output_shapes
:@*
_class
loc:@v/dense/bias
≥
save_1/Assign_24Assignv/dense/bias/Adam_1save_1/RestoreV2:24*
_output_shapes
:@*
use_locking(*
T0*
_class
loc:@v/dense/bias*
validate_shape(
і
save_1/Assign_25Assignv/dense/kernelsave_1/RestoreV2:25*
use_locking(*
T0*!
_class
loc:@v/dense/kernel*
validate_shape(*
_output_shapes

:@
є
save_1/Assign_26Assignv/dense/kernel/Adamsave_1/RestoreV2:26*
validate_shape(*!
_class
loc:@v/dense/kernel*
use_locking(*
T0*
_output_shapes

:@
ї
save_1/Assign_27Assignv/dense/kernel/Adam_1save_1/RestoreV2:27*
_output_shapes

:@*
use_locking(*!
_class
loc:@v/dense/kernel*
T0*
validate_shape(
∞
save_1/Assign_28Assignv/dense_1/biassave_1/RestoreV2:28*
T0*!
_class
loc:@v/dense_1/bias*
validate_shape(*
use_locking(*
_output_shapes
:@
µ
save_1/Assign_29Assignv/dense_1/bias/Adamsave_1/RestoreV2:29*!
_class
loc:@v/dense_1/bias*
validate_shape(*
T0*
use_locking(*
_output_shapes
:@
Ј
save_1/Assign_30Assignv/dense_1/bias/Adam_1save_1/RestoreV2:30*
use_locking(*
T0*!
_class
loc:@v/dense_1/bias*
_output_shapes
:@*
validate_shape(
Є
save_1/Assign_31Assignv/dense_1/kernelsave_1/RestoreV2:31*#
_class
loc:@v/dense_1/kernel*
_output_shapes

:@@*
validate_shape(*
T0*
use_locking(
љ
save_1/Assign_32Assignv/dense_1/kernel/Adamsave_1/RestoreV2:32*
_output_shapes

:@@*#
_class
loc:@v/dense_1/kernel*
use_locking(*
T0*
validate_shape(
њ
save_1/Assign_33Assignv/dense_1/kernel/Adam_1save_1/RestoreV2:33*
validate_shape(*
use_locking(*
T0*
_output_shapes

:@@*#
_class
loc:@v/dense_1/kernel
∞
save_1/Assign_34Assignv/dense_2/biassave_1/RestoreV2:34*
T0*
use_locking(*
_output_shapes
:*!
_class
loc:@v/dense_2/bias*
validate_shape(
µ
save_1/Assign_35Assignv/dense_2/bias/Adamsave_1/RestoreV2:35*!
_class
loc:@v/dense_2/bias*
use_locking(*
_output_shapes
:*
validate_shape(*
T0
Ј
save_1/Assign_36Assignv/dense_2/bias/Adam_1save_1/RestoreV2:36*!
_class
loc:@v/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
Є
save_1/Assign_37Assignv/dense_2/kernelsave_1/RestoreV2:37*
validate_shape(*
_output_shapes

:@*
T0*#
_class
loc:@v/dense_2/kernel*
use_locking(
љ
save_1/Assign_38Assignv/dense_2/kernel/Adamsave_1/RestoreV2:38*
use_locking(*
_output_shapes

:@*
T0*
validate_shape(*#
_class
loc:@v/dense_2/kernel
њ
save_1/Assign_39Assignv/dense_2/kernel/Adam_1save_1/RestoreV2:39*#
_class
loc:@v/dense_2/kernel*
use_locking(*
_output_shapes

:@*
validate_shape(*
T0
И
save_1/restore_shardNoOp^save_1/Assign^save_1/Assign_1^save_1/Assign_10^save_1/Assign_11^save_1/Assign_12^save_1/Assign_13^save_1/Assign_14^save_1/Assign_15^save_1/Assign_16^save_1/Assign_17^save_1/Assign_18^save_1/Assign_19^save_1/Assign_2^save_1/Assign_20^save_1/Assign_21^save_1/Assign_22^save_1/Assign_23^save_1/Assign_24^save_1/Assign_25^save_1/Assign_26^save_1/Assign_27^save_1/Assign_28^save_1/Assign_29^save_1/Assign_3^save_1/Assign_30^save_1/Assign_31^save_1/Assign_32^save_1/Assign_33^save_1/Assign_34^save_1/Assign_35^save_1/Assign_36^save_1/Assign_37^save_1/Assign_38^save_1/Assign_39^save_1/Assign_4^save_1/Assign_5^save_1/Assign_6^save_1/Assign_7^save_1/Assign_8^save_1/Assign_9
1
save_1/restore_allNoOp^save_1/restore_shard
[
save_2/filename/inputConst*
dtype0*
_output_shapes
: *
valueB Bmodel
r
save_2/filenamePlaceholderWithDefaultsave_2/filename/input*
_output_shapes
: *
shape: *
dtype0
i
save_2/ConstPlaceholderWithDefaultsave_2/filename*
dtype0*
_output_shapes
: *
shape: 
Ж
save_2/StringJoin/inputs_1Const*
_output_shapes
: *<
value3B1 B+_temp_933c5e6e3ed14277867b23cedcfd4d95/part*
dtype0
{
save_2/StringJoin
StringJoinsave_2/Constsave_2/StringJoin/inputs_1*
_output_shapes
: *
	separator *
N
S
save_2/num_shardsConst*
dtype0*
_output_shapes
: *
value	B :
^
save_2/ShardedFilename/shardConst*
_output_shapes
: *
dtype0*
value	B : 
Е
save_2/ShardedFilenameShardedFilenamesave_2/StringJoinsave_2/ShardedFilename/shardsave_2/num_shards*
_output_shapes
: 
К
save_2/SaveV2/tensor_namesConst*ї
value±BЃ(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
_output_shapes
:(*
dtype0
µ
save_2/SaveV2/shape_and_slicesConst*
_output_shapes
:(*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
…
save_2/SaveV2SaveV2save_2/ShardedFilenamesave_2/SaveV2/tensor_namessave_2/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1v/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1v/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1v/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1v/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1v/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1v/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1*6
dtypes,
*2(
Щ
save_2/control_dependencyIdentitysave_2/ShardedFilename^save_2/SaveV2*)
_class
loc:@save_2/ShardedFilename*
_output_shapes
: *
T0
£
-save_2/MergeV2Checkpoints/checkpoint_prefixesPacksave_2/ShardedFilename^save_2/control_dependency*
_output_shapes
:*

axis *
N*
T0
Г
save_2/MergeV2CheckpointsMergeV2Checkpoints-save_2/MergeV2Checkpoints/checkpoint_prefixessave_2/Const*
delete_old_dirs(
В
save_2/IdentityIdentitysave_2/Const^save_2/MergeV2Checkpoints^save_2/control_dependency*
T0*
_output_shapes
: 
Н
save_2/RestoreV2/tensor_namesConst*
dtype0*ї
value±BЃ(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
_output_shapes
:(
Є
!save_2/RestoreV2/shape_and_slicesConst*
_output_shapes
:(*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
ё
save_2/RestoreV2	RestoreV2save_2/Constsave_2/RestoreV2/tensor_names!save_2/RestoreV2/shape_and_slices*6
dtypes,
*2(*ґ
_output_shapes£
†::::::::::::::::::::::::::::::::::::::::
Ґ
save_2/AssignAssignbeta1_powersave_2/RestoreV2*
_output_shapes
: *
use_locking(* 
_class
loc:@pi/dense/bias*
validate_shape(*
T0
І
save_2/Assign_1Assignbeta1_power_1save_2/RestoreV2:1*
_output_shapes
: *
_class
loc:@v/dense/bias*
validate_shape(*
use_locking(*
T0
¶
save_2/Assign_2Assignbeta2_powersave_2/RestoreV2:2* 
_class
loc:@pi/dense/bias*
_output_shapes
: *
use_locking(*
validate_shape(*
T0
І
save_2/Assign_3Assignbeta2_power_1save_2/RestoreV2:3*
use_locking(*
T0*
validate_shape(*
_output_shapes
: *
_class
loc:@v/dense/bias
ђ
save_2/Assign_4Assignpi/dense/biassave_2/RestoreV2:4*
T0*
use_locking(*
validate_shape(*
_output_shapes
:@* 
_class
loc:@pi/dense/bias
±
save_2/Assign_5Assignpi/dense/bias/Adamsave_2/RestoreV2:5* 
_class
loc:@pi/dense/bias*
use_locking(*
T0*
validate_shape(*
_output_shapes
:@
≥
save_2/Assign_6Assignpi/dense/bias/Adam_1save_2/RestoreV2:6*
use_locking(*
validate_shape(*
_output_shapes
:@* 
_class
loc:@pi/dense/bias*
T0
і
save_2/Assign_7Assignpi/dense/kernelsave_2/RestoreV2:7*
T0*
use_locking(*"
_class
loc:@pi/dense/kernel*
_output_shapes

:@*
validate_shape(
є
save_2/Assign_8Assignpi/dense/kernel/Adamsave_2/RestoreV2:8*
use_locking(*
validate_shape(*"
_class
loc:@pi/dense/kernel*
T0*
_output_shapes

:@
ї
save_2/Assign_9Assignpi/dense/kernel/Adam_1save_2/RestoreV2:9*
_output_shapes

:@*
use_locking(*
validate_shape(*
T0*"
_class
loc:@pi/dense/kernel
≤
save_2/Assign_10Assignpi/dense_1/biassave_2/RestoreV2:10*
use_locking(*
_output_shapes
:@*
validate_shape(*
T0*"
_class
loc:@pi/dense_1/bias
Ј
save_2/Assign_11Assignpi/dense_1/bias/Adamsave_2/RestoreV2:11*
validate_shape(*
_output_shapes
:@*
T0*
use_locking(*"
_class
loc:@pi/dense_1/bias
є
save_2/Assign_12Assignpi/dense_1/bias/Adam_1save_2/RestoreV2:12*
T0*
validate_shape(*"
_class
loc:@pi/dense_1/bias*
_output_shapes
:@*
use_locking(
Ї
save_2/Assign_13Assignpi/dense_1/kernelsave_2/RestoreV2:13*$
_class
loc:@pi/dense_1/kernel*
use_locking(*
validate_shape(*
_output_shapes

:@@*
T0
њ
save_2/Assign_14Assignpi/dense_1/kernel/Adamsave_2/RestoreV2:14*
T0*$
_class
loc:@pi/dense_1/kernel*
use_locking(*
validate_shape(*
_output_shapes

:@@
Ѕ
save_2/Assign_15Assignpi/dense_1/kernel/Adam_1save_2/RestoreV2:15*
_output_shapes

:@@*
use_locking(*$
_class
loc:@pi/dense_1/kernel*
validate_shape(*
T0
≤
save_2/Assign_16Assignpi/dense_2/biassave_2/RestoreV2:16*
validate_shape(*
use_locking(*"
_class
loc:@pi/dense_2/bias*
T0*
_output_shapes
:
Ј
save_2/Assign_17Assignpi/dense_2/bias/Adamsave_2/RestoreV2:17*
use_locking(*
_output_shapes
:*
validate_shape(*"
_class
loc:@pi/dense_2/bias*
T0
є
save_2/Assign_18Assignpi/dense_2/bias/Adam_1save_2/RestoreV2:18*"
_class
loc:@pi/dense_2/bias*
T0*
validate_shape(*
_output_shapes
:*
use_locking(
Ї
save_2/Assign_19Assignpi/dense_2/kernelsave_2/RestoreV2:19*
use_locking(*
T0*
_output_shapes

:@*
validate_shape(*$
_class
loc:@pi/dense_2/kernel
њ
save_2/Assign_20Assignpi/dense_2/kernel/Adamsave_2/RestoreV2:20*
T0*
validate_shape(*
use_locking(*
_output_shapes

:@*$
_class
loc:@pi/dense_2/kernel
Ѕ
save_2/Assign_21Assignpi/dense_2/kernel/Adam_1save_2/RestoreV2:21*
validate_shape(*$
_class
loc:@pi/dense_2/kernel*
use_locking(*
T0*
_output_shapes

:@
ђ
save_2/Assign_22Assignv/dense/biassave_2/RestoreV2:22*
use_locking(*
T0*
validate_shape(*
_class
loc:@v/dense/bias*
_output_shapes
:@
±
save_2/Assign_23Assignv/dense/bias/Adamsave_2/RestoreV2:23*
use_locking(*
T0*
validate_shape(*
_class
loc:@v/dense/bias*
_output_shapes
:@
≥
save_2/Assign_24Assignv/dense/bias/Adam_1save_2/RestoreV2:24*
_class
loc:@v/dense/bias*
T0*
_output_shapes
:@*
validate_shape(*
use_locking(
і
save_2/Assign_25Assignv/dense/kernelsave_2/RestoreV2:25*
use_locking(*
_output_shapes

:@*
validate_shape(*!
_class
loc:@v/dense/kernel*
T0
є
save_2/Assign_26Assignv/dense/kernel/Adamsave_2/RestoreV2:26*!
_class
loc:@v/dense/kernel*
use_locking(*
validate_shape(*
_output_shapes

:@*
T0
ї
save_2/Assign_27Assignv/dense/kernel/Adam_1save_2/RestoreV2:27*
T0*
use_locking(*
validate_shape(*!
_class
loc:@v/dense/kernel*
_output_shapes

:@
∞
save_2/Assign_28Assignv/dense_1/biassave_2/RestoreV2:28*
T0*
validate_shape(*!
_class
loc:@v/dense_1/bias*
_output_shapes
:@*
use_locking(
µ
save_2/Assign_29Assignv/dense_1/bias/Adamsave_2/RestoreV2:29*!
_class
loc:@v/dense_1/bias*
T0*
use_locking(*
validate_shape(*
_output_shapes
:@
Ј
save_2/Assign_30Assignv/dense_1/bias/Adam_1save_2/RestoreV2:30*
validate_shape(*
_output_shapes
:@*
T0*!
_class
loc:@v/dense_1/bias*
use_locking(
Є
save_2/Assign_31Assignv/dense_1/kernelsave_2/RestoreV2:31*
use_locking(*
T0*
validate_shape(*#
_class
loc:@v/dense_1/kernel*
_output_shapes

:@@
љ
save_2/Assign_32Assignv/dense_1/kernel/Adamsave_2/RestoreV2:32*
validate_shape(*
_output_shapes

:@@*#
_class
loc:@v/dense_1/kernel*
use_locking(*
T0
њ
save_2/Assign_33Assignv/dense_1/kernel/Adam_1save_2/RestoreV2:33*#
_class
loc:@v/dense_1/kernel*
validate_shape(*
T0*
_output_shapes

:@@*
use_locking(
∞
save_2/Assign_34Assignv/dense_2/biassave_2/RestoreV2:34*
T0*
use_locking(*!
_class
loc:@v/dense_2/bias*
validate_shape(*
_output_shapes
:
µ
save_2/Assign_35Assignv/dense_2/bias/Adamsave_2/RestoreV2:35*!
_class
loc:@v/dense_2/bias*
validate_shape(*
use_locking(*
_output_shapes
:*
T0
Ј
save_2/Assign_36Assignv/dense_2/bias/Adam_1save_2/RestoreV2:36*
T0*
_output_shapes
:*!
_class
loc:@v/dense_2/bias*
use_locking(*
validate_shape(
Є
save_2/Assign_37Assignv/dense_2/kernelsave_2/RestoreV2:37*
_output_shapes

:@*#
_class
loc:@v/dense_2/kernel*
use_locking(*
validate_shape(*
T0
љ
save_2/Assign_38Assignv/dense_2/kernel/Adamsave_2/RestoreV2:38*
validate_shape(*#
_class
loc:@v/dense_2/kernel*
use_locking(*
T0*
_output_shapes

:@
њ
save_2/Assign_39Assignv/dense_2/kernel/Adam_1save_2/RestoreV2:39*
T0*
validate_shape(*
use_locking(*#
_class
loc:@v/dense_2/kernel*
_output_shapes

:@
И
save_2/restore_shardNoOp^save_2/Assign^save_2/Assign_1^save_2/Assign_10^save_2/Assign_11^save_2/Assign_12^save_2/Assign_13^save_2/Assign_14^save_2/Assign_15^save_2/Assign_16^save_2/Assign_17^save_2/Assign_18^save_2/Assign_19^save_2/Assign_2^save_2/Assign_20^save_2/Assign_21^save_2/Assign_22^save_2/Assign_23^save_2/Assign_24^save_2/Assign_25^save_2/Assign_26^save_2/Assign_27^save_2/Assign_28^save_2/Assign_29^save_2/Assign_3^save_2/Assign_30^save_2/Assign_31^save_2/Assign_32^save_2/Assign_33^save_2/Assign_34^save_2/Assign_35^save_2/Assign_36^save_2/Assign_37^save_2/Assign_38^save_2/Assign_39^save_2/Assign_4^save_2/Assign_5^save_2/Assign_6^save_2/Assign_7^save_2/Assign_8^save_2/Assign_9
1
save_2/restore_allNoOp^save_2/restore_shard
[
save_3/filename/inputConst*
valueB Bmodel*
_output_shapes
: *
dtype0
r
save_3/filenamePlaceholderWithDefaultsave_3/filename/input*
shape: *
dtype0*
_output_shapes
: 
i
save_3/ConstPlaceholderWithDefaultsave_3/filename*
_output_shapes
: *
shape: *
dtype0
Ж
save_3/StringJoin/inputs_1Const*<
value3B1 B+_temp_0b9f082150c54a0e8cb106b9aee215e1/part*
dtype0*
_output_shapes
: 
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
dtype0*
value	B : *
_output_shapes
: 
Е
save_3/ShardedFilenameShardedFilenamesave_3/StringJoinsave_3/ShardedFilename/shardsave_3/num_shards*
_output_shapes
: 
К
save_3/SaveV2/tensor_namesConst*
dtype0*ї
value±BЃ(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
_output_shapes
:(
µ
save_3/SaveV2/shape_and_slicesConst*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:(*
dtype0
…
save_3/SaveV2SaveV2save_3/ShardedFilenamesave_3/SaveV2/tensor_namessave_3/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1v/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1v/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1v/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1v/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1v/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1v/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1*6
dtypes,
*2(
Щ
save_3/control_dependencyIdentitysave_3/ShardedFilename^save_3/SaveV2*
T0*)
_class
loc:@save_3/ShardedFilename*
_output_shapes
: 
£
-save_3/MergeV2Checkpoints/checkpoint_prefixesPacksave_3/ShardedFilename^save_3/control_dependency*
T0*
_output_shapes
:*
N*

axis 
Г
save_3/MergeV2CheckpointsMergeV2Checkpoints-save_3/MergeV2Checkpoints/checkpoint_prefixessave_3/Const*
delete_old_dirs(
В
save_3/IdentityIdentitysave_3/Const^save_3/MergeV2Checkpoints^save_3/control_dependency*
_output_shapes
: *
T0
Н
save_3/RestoreV2/tensor_namesConst*
dtype0*
_output_shapes
:(*ї
value±BЃ(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1
Є
!save_3/RestoreV2/shape_and_slicesConst*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:(*
dtype0
ё
save_3/RestoreV2	RestoreV2save_3/Constsave_3/RestoreV2/tensor_names!save_3/RestoreV2/shape_and_slices*ґ
_output_shapes£
†::::::::::::::::::::::::::::::::::::::::*6
dtypes,
*2(
Ґ
save_3/AssignAssignbeta1_powersave_3/RestoreV2*
_output_shapes
: *
validate_shape(* 
_class
loc:@pi/dense/bias*
use_locking(*
T0
І
save_3/Assign_1Assignbeta1_power_1save_3/RestoreV2:1*
validate_shape(*
T0*
_output_shapes
: *
_class
loc:@v/dense/bias*
use_locking(
¶
save_3/Assign_2Assignbeta2_powersave_3/RestoreV2:2*
validate_shape(*
use_locking(*
_output_shapes
: *
T0* 
_class
loc:@pi/dense/bias
І
save_3/Assign_3Assignbeta2_power_1save_3/RestoreV2:3*
_output_shapes
: *
_class
loc:@v/dense/bias*
validate_shape(*
T0*
use_locking(
ђ
save_3/Assign_4Assignpi/dense/biassave_3/RestoreV2:4*
validate_shape(* 
_class
loc:@pi/dense/bias*
T0*
_output_shapes
:@*
use_locking(
±
save_3/Assign_5Assignpi/dense/bias/Adamsave_3/RestoreV2:5*
T0*
_output_shapes
:@* 
_class
loc:@pi/dense/bias*
validate_shape(*
use_locking(
≥
save_3/Assign_6Assignpi/dense/bias/Adam_1save_3/RestoreV2:6* 
_class
loc:@pi/dense/bias*
use_locking(*
_output_shapes
:@*
validate_shape(*
T0
і
save_3/Assign_7Assignpi/dense/kernelsave_3/RestoreV2:7*"
_class
loc:@pi/dense/kernel*
_output_shapes

:@*
use_locking(*
validate_shape(*
T0
є
save_3/Assign_8Assignpi/dense/kernel/Adamsave_3/RestoreV2:8*
_output_shapes

:@*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(
ї
save_3/Assign_9Assignpi/dense/kernel/Adam_1save_3/RestoreV2:9*"
_class
loc:@pi/dense/kernel*
use_locking(*
T0*
validate_shape(*
_output_shapes

:@
≤
save_3/Assign_10Assignpi/dense_1/biassave_3/RestoreV2:10*
_output_shapes
:@*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
T0*
use_locking(
Ј
save_3/Assign_11Assignpi/dense_1/bias/Adamsave_3/RestoreV2:11*
use_locking(*
_output_shapes
:@*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
T0
є
save_3/Assign_12Assignpi/dense_1/bias/Adam_1save_3/RestoreV2:12*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
_output_shapes
:@*
T0*
use_locking(
Ї
save_3/Assign_13Assignpi/dense_1/kernelsave_3/RestoreV2:13*
_output_shapes

:@@*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel*
validate_shape(
њ
save_3/Assign_14Assignpi/dense_1/kernel/Adamsave_3/RestoreV2:14*$
_class
loc:@pi/dense_1/kernel*
use_locking(*
T0*
_output_shapes

:@@*
validate_shape(
Ѕ
save_3/Assign_15Assignpi/dense_1/kernel/Adam_1save_3/RestoreV2:15*
use_locking(*
_output_shapes

:@@*
T0*$
_class
loc:@pi/dense_1/kernel*
validate_shape(
≤
save_3/Assign_16Assignpi/dense_2/biassave_3/RestoreV2:16*
T0*"
_class
loc:@pi/dense_2/bias*
use_locking(*
validate_shape(*
_output_shapes
:
Ј
save_3/Assign_17Assignpi/dense_2/bias/Adamsave_3/RestoreV2:17*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:*
validate_shape(
є
save_3/Assign_18Assignpi/dense_2/bias/Adam_1save_3/RestoreV2:18*
T0*"
_class
loc:@pi/dense_2/bias*
use_locking(*
validate_shape(*
_output_shapes
:
Ї
save_3/Assign_19Assignpi/dense_2/kernelsave_3/RestoreV2:19*
validate_shape(*
_output_shapes

:@*
use_locking(*$
_class
loc:@pi/dense_2/kernel*
T0
њ
save_3/Assign_20Assignpi/dense_2/kernel/Adamsave_3/RestoreV2:20*$
_class
loc:@pi/dense_2/kernel*
use_locking(*
T0*
_output_shapes

:@*
validate_shape(
Ѕ
save_3/Assign_21Assignpi/dense_2/kernel/Adam_1save_3/RestoreV2:21*$
_class
loc:@pi/dense_2/kernel*
_output_shapes

:@*
T0*
validate_shape(*
use_locking(
ђ
save_3/Assign_22Assignv/dense/biassave_3/RestoreV2:22*
validate_shape(*
_output_shapes
:@*
T0*
use_locking(*
_class
loc:@v/dense/bias
±
save_3/Assign_23Assignv/dense/bias/Adamsave_3/RestoreV2:23*
use_locking(*
_output_shapes
:@*
_class
loc:@v/dense/bias*
T0*
validate_shape(
≥
save_3/Assign_24Assignv/dense/bias/Adam_1save_3/RestoreV2:24*
validate_shape(*
T0*
_output_shapes
:@*
use_locking(*
_class
loc:@v/dense/bias
і
save_3/Assign_25Assignv/dense/kernelsave_3/RestoreV2:25*
use_locking(*!
_class
loc:@v/dense/kernel*
validate_shape(*
_output_shapes

:@*
T0
є
save_3/Assign_26Assignv/dense/kernel/Adamsave_3/RestoreV2:26*
_output_shapes

:@*
T0*
validate_shape(*!
_class
loc:@v/dense/kernel*
use_locking(
ї
save_3/Assign_27Assignv/dense/kernel/Adam_1save_3/RestoreV2:27*
use_locking(*
_output_shapes

:@*
T0*
validate_shape(*!
_class
loc:@v/dense/kernel
∞
save_3/Assign_28Assignv/dense_1/biassave_3/RestoreV2:28*
validate_shape(*
T0*
use_locking(*!
_class
loc:@v/dense_1/bias*
_output_shapes
:@
µ
save_3/Assign_29Assignv/dense_1/bias/Adamsave_3/RestoreV2:29*!
_class
loc:@v/dense_1/bias*
T0*
_output_shapes
:@*
use_locking(*
validate_shape(
Ј
save_3/Assign_30Assignv/dense_1/bias/Adam_1save_3/RestoreV2:30*
_output_shapes
:@*
validate_shape(*!
_class
loc:@v/dense_1/bias*
use_locking(*
T0
Є
save_3/Assign_31Assignv/dense_1/kernelsave_3/RestoreV2:31*#
_class
loc:@v/dense_1/kernel*
validate_shape(*
use_locking(*
_output_shapes

:@@*
T0
љ
save_3/Assign_32Assignv/dense_1/kernel/Adamsave_3/RestoreV2:32*
use_locking(*
_output_shapes

:@@*
T0*
validate_shape(*#
_class
loc:@v/dense_1/kernel
њ
save_3/Assign_33Assignv/dense_1/kernel/Adam_1save_3/RestoreV2:33*
use_locking(*
_output_shapes

:@@*#
_class
loc:@v/dense_1/kernel*
T0*
validate_shape(
∞
save_3/Assign_34Assignv/dense_2/biassave_3/RestoreV2:34*
T0*
validate_shape(*
use_locking(*
_output_shapes
:*!
_class
loc:@v/dense_2/bias
µ
save_3/Assign_35Assignv/dense_2/bias/Adamsave_3/RestoreV2:35*
use_locking(*
validate_shape(*
T0*!
_class
loc:@v/dense_2/bias*
_output_shapes
:
Ј
save_3/Assign_36Assignv/dense_2/bias/Adam_1save_3/RestoreV2:36*
_output_shapes
:*
use_locking(*!
_class
loc:@v/dense_2/bias*
validate_shape(*
T0
Є
save_3/Assign_37Assignv/dense_2/kernelsave_3/RestoreV2:37*
use_locking(*
T0*#
_class
loc:@v/dense_2/kernel*
_output_shapes

:@*
validate_shape(
љ
save_3/Assign_38Assignv/dense_2/kernel/Adamsave_3/RestoreV2:38*
_output_shapes

:@*#
_class
loc:@v/dense_2/kernel*
use_locking(*
validate_shape(*
T0
њ
save_3/Assign_39Assignv/dense_2/kernel/Adam_1save_3/RestoreV2:39*#
_class
loc:@v/dense_2/kernel*
validate_shape(*
T0*
_output_shapes

:@*
use_locking(
И
save_3/restore_shardNoOp^save_3/Assign^save_3/Assign_1^save_3/Assign_10^save_3/Assign_11^save_3/Assign_12^save_3/Assign_13^save_3/Assign_14^save_3/Assign_15^save_3/Assign_16^save_3/Assign_17^save_3/Assign_18^save_3/Assign_19^save_3/Assign_2^save_3/Assign_20^save_3/Assign_21^save_3/Assign_22^save_3/Assign_23^save_3/Assign_24^save_3/Assign_25^save_3/Assign_26^save_3/Assign_27^save_3/Assign_28^save_3/Assign_29^save_3/Assign_3^save_3/Assign_30^save_3/Assign_31^save_3/Assign_32^save_3/Assign_33^save_3/Assign_34^save_3/Assign_35^save_3/Assign_36^save_3/Assign_37^save_3/Assign_38^save_3/Assign_39^save_3/Assign_4^save_3/Assign_5^save_3/Assign_6^save_3/Assign_7^save_3/Assign_8^save_3/Assign_9
1
save_3/restore_allNoOp^save_3/restore_shard
[
save_4/filename/inputConst*
_output_shapes
: *
dtype0*
valueB Bmodel
r
save_4/filenamePlaceholderWithDefaultsave_4/filename/input*
shape: *
_output_shapes
: *
dtype0
i
save_4/ConstPlaceholderWithDefaultsave_4/filename*
shape: *
_output_shapes
: *
dtype0
Ж
save_4/StringJoin/inputs_1Const*
dtype0*<
value3B1 B+_temp_c42e9579bc19442e963e70d61dd25349/part*
_output_shapes
: 
{
save_4/StringJoin
StringJoinsave_4/Constsave_4/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
S
save_4/num_shardsConst*
_output_shapes
: *
dtype0*
value	B :
^
save_4/ShardedFilename/shardConst*
dtype0*
_output_shapes
: *
value	B : 
Е
save_4/ShardedFilenameShardedFilenamesave_4/StringJoinsave_4/ShardedFilename/shardsave_4/num_shards*
_output_shapes
: 
К
save_4/SaveV2/tensor_namesConst*ї
value±BЃ(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
_output_shapes
:(*
dtype0
µ
save_4/SaveV2/shape_and_slicesConst*
dtype0*
_output_shapes
:(*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
…
save_4/SaveV2SaveV2save_4/ShardedFilenamesave_4/SaveV2/tensor_namessave_4/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1v/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1v/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1v/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1v/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1v/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1v/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1*6
dtypes,
*2(
Щ
save_4/control_dependencyIdentitysave_4/ShardedFilename^save_4/SaveV2*
T0*)
_class
loc:@save_4/ShardedFilename*
_output_shapes
: 
£
-save_4/MergeV2Checkpoints/checkpoint_prefixesPacksave_4/ShardedFilename^save_4/control_dependency*
_output_shapes
:*
T0*
N*

axis 
Г
save_4/MergeV2CheckpointsMergeV2Checkpoints-save_4/MergeV2Checkpoints/checkpoint_prefixessave_4/Const*
delete_old_dirs(
В
save_4/IdentityIdentitysave_4/Const^save_4/MergeV2Checkpoints^save_4/control_dependency*
T0*
_output_shapes
: 
Н
save_4/RestoreV2/tensor_namesConst*
_output_shapes
:(*
dtype0*ї
value±BЃ(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1
Є
!save_4/RestoreV2/shape_and_slicesConst*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:(
ё
save_4/RestoreV2	RestoreV2save_4/Constsave_4/RestoreV2/tensor_names!save_4/RestoreV2/shape_and_slices*ґ
_output_shapes£
†::::::::::::::::::::::::::::::::::::::::*6
dtypes,
*2(
Ґ
save_4/AssignAssignbeta1_powersave_4/RestoreV2* 
_class
loc:@pi/dense/bias*
validate_shape(*
use_locking(*
T0*
_output_shapes
: 
І
save_4/Assign_1Assignbeta1_power_1save_4/RestoreV2:1*
_output_shapes
: *
T0*
validate_shape(*
use_locking(*
_class
loc:@v/dense/bias
¶
save_4/Assign_2Assignbeta2_powersave_4/RestoreV2:2*
_output_shapes
: * 
_class
loc:@pi/dense/bias*
use_locking(*
T0*
validate_shape(
І
save_4/Assign_3Assignbeta2_power_1save_4/RestoreV2:3*
T0*
validate_shape(*
_class
loc:@v/dense/bias*
_output_shapes
: *
use_locking(
ђ
save_4/Assign_4Assignpi/dense/biassave_4/RestoreV2:4*
T0*
_output_shapes
:@*
validate_shape(*
use_locking(* 
_class
loc:@pi/dense/bias
±
save_4/Assign_5Assignpi/dense/bias/Adamsave_4/RestoreV2:5*
validate_shape(*
_output_shapes
:@*
T0* 
_class
loc:@pi/dense/bias*
use_locking(
≥
save_4/Assign_6Assignpi/dense/bias/Adam_1save_4/RestoreV2:6*
validate_shape(* 
_class
loc:@pi/dense/bias*
use_locking(*
T0*
_output_shapes
:@
і
save_4/Assign_7Assignpi/dense/kernelsave_4/RestoreV2:7*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel*
_output_shapes

:@*
validate_shape(
є
save_4/Assign_8Assignpi/dense/kernel/Adamsave_4/RestoreV2:8*
_output_shapes

:@*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(*
use_locking(
ї
save_4/Assign_9Assignpi/dense/kernel/Adam_1save_4/RestoreV2:9*
validate_shape(*
use_locking(*"
_class
loc:@pi/dense/kernel*
T0*
_output_shapes

:@
≤
save_4/Assign_10Assignpi/dense_1/biassave_4/RestoreV2:10*
_output_shapes
:@*
validate_shape(*"
_class
loc:@pi/dense_1/bias*
use_locking(*
T0
Ј
save_4/Assign_11Assignpi/dense_1/bias/Adamsave_4/RestoreV2:11*"
_class
loc:@pi/dense_1/bias*
_output_shapes
:@*
T0*
use_locking(*
validate_shape(
є
save_4/Assign_12Assignpi/dense_1/bias/Adam_1save_4/RestoreV2:12*
T0*
_output_shapes
:@*
validate_shape(*"
_class
loc:@pi/dense_1/bias*
use_locking(
Ї
save_4/Assign_13Assignpi/dense_1/kernelsave_4/RestoreV2:13*$
_class
loc:@pi/dense_1/kernel*
validate_shape(*
T0*
use_locking(*
_output_shapes

:@@
њ
save_4/Assign_14Assignpi/dense_1/kernel/Adamsave_4/RestoreV2:14*
validate_shape(*$
_class
loc:@pi/dense_1/kernel*
_output_shapes

:@@*
T0*
use_locking(
Ѕ
save_4/Assign_15Assignpi/dense_1/kernel/Adam_1save_4/RestoreV2:15*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
≤
save_4/Assign_16Assignpi/dense_2/biassave_4/RestoreV2:16*
_output_shapes
:*
validate_shape(*
T0*"
_class
loc:@pi/dense_2/bias*
use_locking(
Ј
save_4/Assign_17Assignpi/dense_2/bias/Adamsave_4/RestoreV2:17*
use_locking(*
validate_shape(*"
_class
loc:@pi/dense_2/bias*
T0*
_output_shapes
:
є
save_4/Assign_18Assignpi/dense_2/bias/Adam_1save_4/RestoreV2:18*
T0*
_output_shapes
:*
validate_shape(*"
_class
loc:@pi/dense_2/bias*
use_locking(
Ї
save_4/Assign_19Assignpi/dense_2/kernelsave_4/RestoreV2:19*$
_class
loc:@pi/dense_2/kernel*
use_locking(*
T0*
_output_shapes

:@*
validate_shape(
њ
save_4/Assign_20Assignpi/dense_2/kernel/Adamsave_4/RestoreV2:20*
use_locking(*$
_class
loc:@pi/dense_2/kernel*
T0*
_output_shapes

:@*
validate_shape(
Ѕ
save_4/Assign_21Assignpi/dense_2/kernel/Adam_1save_4/RestoreV2:21*$
_class
loc:@pi/dense_2/kernel*
T0*
_output_shapes

:@*
use_locking(*
validate_shape(
ђ
save_4/Assign_22Assignv/dense/biassave_4/RestoreV2:22*
validate_shape(*
T0*
use_locking(*
_output_shapes
:@*
_class
loc:@v/dense/bias
±
save_4/Assign_23Assignv/dense/bias/Adamsave_4/RestoreV2:23*
_class
loc:@v/dense/bias*
validate_shape(*
use_locking(*
_output_shapes
:@*
T0
≥
save_4/Assign_24Assignv/dense/bias/Adam_1save_4/RestoreV2:24*
T0*
_output_shapes
:@*
validate_shape(*
_class
loc:@v/dense/bias*
use_locking(
і
save_4/Assign_25Assignv/dense/kernelsave_4/RestoreV2:25*
_output_shapes

:@*
validate_shape(*!
_class
loc:@v/dense/kernel*
use_locking(*
T0
є
save_4/Assign_26Assignv/dense/kernel/Adamsave_4/RestoreV2:26*
T0*
_output_shapes

:@*!
_class
loc:@v/dense/kernel*
validate_shape(*
use_locking(
ї
save_4/Assign_27Assignv/dense/kernel/Adam_1save_4/RestoreV2:27*!
_class
loc:@v/dense/kernel*
use_locking(*
_output_shapes

:@*
T0*
validate_shape(
∞
save_4/Assign_28Assignv/dense_1/biassave_4/RestoreV2:28*!
_class
loc:@v/dense_1/bias*
use_locking(*
validate_shape(*
T0*
_output_shapes
:@
µ
save_4/Assign_29Assignv/dense_1/bias/Adamsave_4/RestoreV2:29*
T0*
_output_shapes
:@*
use_locking(*
validate_shape(*!
_class
loc:@v/dense_1/bias
Ј
save_4/Assign_30Assignv/dense_1/bias/Adam_1save_4/RestoreV2:30*
_output_shapes
:@*
validate_shape(*!
_class
loc:@v/dense_1/bias*
use_locking(*
T0
Є
save_4/Assign_31Assignv/dense_1/kernelsave_4/RestoreV2:31*
T0*
use_locking(*
_output_shapes

:@@*#
_class
loc:@v/dense_1/kernel*
validate_shape(
љ
save_4/Assign_32Assignv/dense_1/kernel/Adamsave_4/RestoreV2:32*#
_class
loc:@v/dense_1/kernel*
validate_shape(*
_output_shapes

:@@*
T0*
use_locking(
њ
save_4/Assign_33Assignv/dense_1/kernel/Adam_1save_4/RestoreV2:33*
_output_shapes

:@@*#
_class
loc:@v/dense_1/kernel*
use_locking(*
validate_shape(*
T0
∞
save_4/Assign_34Assignv/dense_2/biassave_4/RestoreV2:34*
use_locking(*!
_class
loc:@v/dense_2/bias*
_output_shapes
:*
validate_shape(*
T0
µ
save_4/Assign_35Assignv/dense_2/bias/Adamsave_4/RestoreV2:35*
T0*
_output_shapes
:*
use_locking(*
validate_shape(*!
_class
loc:@v/dense_2/bias
Ј
save_4/Assign_36Assignv/dense_2/bias/Adam_1save_4/RestoreV2:36*
_output_shapes
:*
use_locking(*!
_class
loc:@v/dense_2/bias*
validate_shape(*
T0
Є
save_4/Assign_37Assignv/dense_2/kernelsave_4/RestoreV2:37*
T0*
validate_shape(*#
_class
loc:@v/dense_2/kernel*
use_locking(*
_output_shapes

:@
љ
save_4/Assign_38Assignv/dense_2/kernel/Adamsave_4/RestoreV2:38*#
_class
loc:@v/dense_2/kernel*
_output_shapes

:@*
validate_shape(*
T0*
use_locking(
њ
save_4/Assign_39Assignv/dense_2/kernel/Adam_1save_4/RestoreV2:39*
T0*
validate_shape(*
use_locking(*#
_class
loc:@v/dense_2/kernel*
_output_shapes

:@
И
save_4/restore_shardNoOp^save_4/Assign^save_4/Assign_1^save_4/Assign_10^save_4/Assign_11^save_4/Assign_12^save_4/Assign_13^save_4/Assign_14^save_4/Assign_15^save_4/Assign_16^save_4/Assign_17^save_4/Assign_18^save_4/Assign_19^save_4/Assign_2^save_4/Assign_20^save_4/Assign_21^save_4/Assign_22^save_4/Assign_23^save_4/Assign_24^save_4/Assign_25^save_4/Assign_26^save_4/Assign_27^save_4/Assign_28^save_4/Assign_29^save_4/Assign_3^save_4/Assign_30^save_4/Assign_31^save_4/Assign_32^save_4/Assign_33^save_4/Assign_34^save_4/Assign_35^save_4/Assign_36^save_4/Assign_37^save_4/Assign_38^save_4/Assign_39^save_4/Assign_4^save_4/Assign_5^save_4/Assign_6^save_4/Assign_7^save_4/Assign_8^save_4/Assign_9
1
save_4/restore_allNoOp^save_4/restore_shard
[
save_5/filename/inputConst*
dtype0*
_output_shapes
: *
valueB Bmodel
r
save_5/filenamePlaceholderWithDefaultsave_5/filename/input*
shape: *
_output_shapes
: *
dtype0
i
save_5/ConstPlaceholderWithDefaultsave_5/filename*
_output_shapes
: *
shape: *
dtype0
Ж
save_5/StringJoin/inputs_1Const*
_output_shapes
: *<
value3B1 B+_temp_ec8b41904d5c48a584548db9bd72f041/part*
dtype0
{
save_5/StringJoin
StringJoinsave_5/Constsave_5/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
S
save_5/num_shardsConst*
dtype0*
_output_shapes
: *
value	B :
^
save_5/ShardedFilename/shardConst*
value	B : *
_output_shapes
: *
dtype0
Е
save_5/ShardedFilenameShardedFilenamesave_5/StringJoinsave_5/ShardedFilename/shardsave_5/num_shards*
_output_shapes
: 
К
save_5/SaveV2/tensor_namesConst*ї
value±BЃ(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:(
µ
save_5/SaveV2/shape_and_slicesConst*
_output_shapes
:(*
dtype0*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
…
save_5/SaveV2SaveV2save_5/ShardedFilenamesave_5/SaveV2/tensor_namessave_5/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1v/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1v/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1v/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1v/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1v/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1v/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1*6
dtypes,
*2(
Щ
save_5/control_dependencyIdentitysave_5/ShardedFilename^save_5/SaveV2*)
_class
loc:@save_5/ShardedFilename*
_output_shapes
: *
T0
£
-save_5/MergeV2Checkpoints/checkpoint_prefixesPacksave_5/ShardedFilename^save_5/control_dependency*
N*

axis *
_output_shapes
:*
T0
Г
save_5/MergeV2CheckpointsMergeV2Checkpoints-save_5/MergeV2Checkpoints/checkpoint_prefixessave_5/Const*
delete_old_dirs(
В
save_5/IdentityIdentitysave_5/Const^save_5/MergeV2Checkpoints^save_5/control_dependency*
T0*
_output_shapes
: 
Н
save_5/RestoreV2/tensor_namesConst*ї
value±BЃ(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
_output_shapes
:(*
dtype0
Є
!save_5/RestoreV2/shape_and_slicesConst*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:(
ё
save_5/RestoreV2	RestoreV2save_5/Constsave_5/RestoreV2/tensor_names!save_5/RestoreV2/shape_and_slices*ґ
_output_shapes£
†::::::::::::::::::::::::::::::::::::::::*6
dtypes,
*2(
Ґ
save_5/AssignAssignbeta1_powersave_5/RestoreV2*
_output_shapes
: * 
_class
loc:@pi/dense/bias*
T0*
validate_shape(*
use_locking(
І
save_5/Assign_1Assignbeta1_power_1save_5/RestoreV2:1*
_class
loc:@v/dense/bias*
T0*
use_locking(*
_output_shapes
: *
validate_shape(
¶
save_5/Assign_2Assignbeta2_powersave_5/RestoreV2:2*
use_locking(*
validate_shape(*
_output_shapes
: * 
_class
loc:@pi/dense/bias*
T0
І
save_5/Assign_3Assignbeta2_power_1save_5/RestoreV2:3*
use_locking(*
_class
loc:@v/dense/bias*
T0*
_output_shapes
: *
validate_shape(
ђ
save_5/Assign_4Assignpi/dense/biassave_5/RestoreV2:4*
T0*
use_locking(* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
:@
±
save_5/Assign_5Assignpi/dense/bias/Adamsave_5/RestoreV2:5* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0
≥
save_5/Assign_6Assignpi/dense/bias/Adam_1save_5/RestoreV2:6*
validate_shape(* 
_class
loc:@pi/dense/bias*
T0*
_output_shapes
:@*
use_locking(
і
save_5/Assign_7Assignpi/dense/kernelsave_5/RestoreV2:7*
T0*"
_class
loc:@pi/dense/kernel*
use_locking(*
validate_shape(*
_output_shapes

:@
є
save_5/Assign_8Assignpi/dense/kernel/Adamsave_5/RestoreV2:8*
validate_shape(*
T0*"
_class
loc:@pi/dense/kernel*
use_locking(*
_output_shapes

:@
ї
save_5/Assign_9Assignpi/dense/kernel/Adam_1save_5/RestoreV2:9*
validate_shape(*
use_locking(*"
_class
loc:@pi/dense/kernel*
_output_shapes

:@*
T0
≤
save_5/Assign_10Assignpi/dense_1/biassave_5/RestoreV2:10*
_output_shapes
:@*
T0*
validate_shape(*"
_class
loc:@pi/dense_1/bias*
use_locking(
Ј
save_5/Assign_11Assignpi/dense_1/bias/Adamsave_5/RestoreV2:11*
validate_shape(*
T0*
use_locking(*
_output_shapes
:@*"
_class
loc:@pi/dense_1/bias
є
save_5/Assign_12Assignpi/dense_1/bias/Adam_1save_5/RestoreV2:12*
use_locking(*
_output_shapes
:@*"
_class
loc:@pi/dense_1/bias*
T0*
validate_shape(
Ї
save_5/Assign_13Assignpi/dense_1/kernelsave_5/RestoreV2:13*
T0*
use_locking(*
_output_shapes

:@@*$
_class
loc:@pi/dense_1/kernel*
validate_shape(
њ
save_5/Assign_14Assignpi/dense_1/kernel/Adamsave_5/RestoreV2:14*$
_class
loc:@pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@*
T0*
use_locking(
Ѕ
save_5/Assign_15Assignpi/dense_1/kernel/Adam_1save_5/RestoreV2:15*
validate_shape(*
T0*
use_locking(*$
_class
loc:@pi/dense_1/kernel*
_output_shapes

:@@
≤
save_5/Assign_16Assignpi/dense_2/biassave_5/RestoreV2:16*
T0*
use_locking(*
_output_shapes
:*
validate_shape(*"
_class
loc:@pi/dense_2/bias
Ј
save_5/Assign_17Assignpi/dense_2/bias/Adamsave_5/RestoreV2:17*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
use_locking(*
T0*
_output_shapes
:
є
save_5/Assign_18Assignpi/dense_2/bias/Adam_1save_5/RestoreV2:18*
use_locking(*
validate_shape(*
_output_shapes
:*"
_class
loc:@pi/dense_2/bias*
T0
Ї
save_5/Assign_19Assignpi/dense_2/kernelsave_5/RestoreV2:19*$
_class
loc:@pi/dense_2/kernel*
_output_shapes

:@*
validate_shape(*
T0*
use_locking(
њ
save_5/Assign_20Assignpi/dense_2/kernel/Adamsave_5/RestoreV2:20*
T0*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
use_locking(*
_output_shapes

:@
Ѕ
save_5/Assign_21Assignpi/dense_2/kernel/Adam_1save_5/RestoreV2:21*$
_class
loc:@pi/dense_2/kernel*
use_locking(*
validate_shape(*
T0*
_output_shapes

:@
ђ
save_5/Assign_22Assignv/dense/biassave_5/RestoreV2:22*
_class
loc:@v/dense/bias*
use_locking(*
T0*
_output_shapes
:@*
validate_shape(
±
save_5/Assign_23Assignv/dense/bias/Adamsave_5/RestoreV2:23*
_class
loc:@v/dense/bias*
_output_shapes
:@*
use_locking(*
T0*
validate_shape(
≥
save_5/Assign_24Assignv/dense/bias/Adam_1save_5/RestoreV2:24*
T0*
validate_shape(*
_output_shapes
:@*
_class
loc:@v/dense/bias*
use_locking(
і
save_5/Assign_25Assignv/dense/kernelsave_5/RestoreV2:25*
validate_shape(*
T0*
use_locking(*
_output_shapes

:@*!
_class
loc:@v/dense/kernel
є
save_5/Assign_26Assignv/dense/kernel/Adamsave_5/RestoreV2:26*
T0*
validate_shape(*
use_locking(*!
_class
loc:@v/dense/kernel*
_output_shapes

:@
ї
save_5/Assign_27Assignv/dense/kernel/Adam_1save_5/RestoreV2:27*
T0*
_output_shapes

:@*
use_locking(*!
_class
loc:@v/dense/kernel*
validate_shape(
∞
save_5/Assign_28Assignv/dense_1/biassave_5/RestoreV2:28*
_output_shapes
:@*
use_locking(*
T0*
validate_shape(*!
_class
loc:@v/dense_1/bias
µ
save_5/Assign_29Assignv/dense_1/bias/Adamsave_5/RestoreV2:29*
use_locking(*
T0*
_output_shapes
:@*!
_class
loc:@v/dense_1/bias*
validate_shape(
Ј
save_5/Assign_30Assignv/dense_1/bias/Adam_1save_5/RestoreV2:30*!
_class
loc:@v/dense_1/bias*
use_locking(*
_output_shapes
:@*
validate_shape(*
T0
Є
save_5/Assign_31Assignv/dense_1/kernelsave_5/RestoreV2:31*
validate_shape(*
T0*#
_class
loc:@v/dense_1/kernel*
use_locking(*
_output_shapes

:@@
љ
save_5/Assign_32Assignv/dense_1/kernel/Adamsave_5/RestoreV2:32*#
_class
loc:@v/dense_1/kernel*
T0*
validate_shape(*
use_locking(*
_output_shapes

:@@
њ
save_5/Assign_33Assignv/dense_1/kernel/Adam_1save_5/RestoreV2:33*
T0*
_output_shapes

:@@*
validate_shape(*
use_locking(*#
_class
loc:@v/dense_1/kernel
∞
save_5/Assign_34Assignv/dense_2/biassave_5/RestoreV2:34*
_output_shapes
:*
T0*
use_locking(*!
_class
loc:@v/dense_2/bias*
validate_shape(
µ
save_5/Assign_35Assignv/dense_2/bias/Adamsave_5/RestoreV2:35*!
_class
loc:@v/dense_2/bias*
T0*
use_locking(*
validate_shape(*
_output_shapes
:
Ј
save_5/Assign_36Assignv/dense_2/bias/Adam_1save_5/RestoreV2:36*
validate_shape(*
use_locking(*
_output_shapes
:*
T0*!
_class
loc:@v/dense_2/bias
Є
save_5/Assign_37Assignv/dense_2/kernelsave_5/RestoreV2:37*
_output_shapes

:@*#
_class
loc:@v/dense_2/kernel*
validate_shape(*
use_locking(*
T0
љ
save_5/Assign_38Assignv/dense_2/kernel/Adamsave_5/RestoreV2:38*#
_class
loc:@v/dense_2/kernel*
validate_shape(*
use_locking(*
_output_shapes

:@*
T0
њ
save_5/Assign_39Assignv/dense_2/kernel/Adam_1save_5/RestoreV2:39*#
_class
loc:@v/dense_2/kernel*
T0*
use_locking(*
_output_shapes

:@*
validate_shape(
И
save_5/restore_shardNoOp^save_5/Assign^save_5/Assign_1^save_5/Assign_10^save_5/Assign_11^save_5/Assign_12^save_5/Assign_13^save_5/Assign_14^save_5/Assign_15^save_5/Assign_16^save_5/Assign_17^save_5/Assign_18^save_5/Assign_19^save_5/Assign_2^save_5/Assign_20^save_5/Assign_21^save_5/Assign_22^save_5/Assign_23^save_5/Assign_24^save_5/Assign_25^save_5/Assign_26^save_5/Assign_27^save_5/Assign_28^save_5/Assign_29^save_5/Assign_3^save_5/Assign_30^save_5/Assign_31^save_5/Assign_32^save_5/Assign_33^save_5/Assign_34^save_5/Assign_35^save_5/Assign_36^save_5/Assign_37^save_5/Assign_38^save_5/Assign_39^save_5/Assign_4^save_5/Assign_5^save_5/Assign_6^save_5/Assign_7^save_5/Assign_8^save_5/Assign_9
1
save_5/restore_allNoOp^save_5/restore_shard
[
save_6/filename/inputConst*
_output_shapes
: *
dtype0*
valueB Bmodel
r
save_6/filenamePlaceholderWithDefaultsave_6/filename/input*
_output_shapes
: *
shape: *
dtype0
i
save_6/ConstPlaceholderWithDefaultsave_6/filename*
_output_shapes
: *
shape: *
dtype0
Ж
save_6/StringJoin/inputs_1Const*<
value3B1 B+_temp_5869bcea8aaf44f89b894fbf01d60945/part*
dtype0*
_output_shapes
: 
{
save_6/StringJoin
StringJoinsave_6/Constsave_6/StringJoin/inputs_1*
	separator *
_output_shapes
: *
N
S
save_6/num_shardsConst*
_output_shapes
: *
dtype0*
value	B :
^
save_6/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 
Е
save_6/ShardedFilenameShardedFilenamesave_6/StringJoinsave_6/ShardedFilename/shardsave_6/num_shards*
_output_shapes
: 
К
save_6/SaveV2/tensor_namesConst*
_output_shapes
:(*
dtype0*ї
value±BЃ(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1
µ
save_6/SaveV2/shape_and_slicesConst*
dtype0*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:(
…
save_6/SaveV2SaveV2save_6/ShardedFilenamesave_6/SaveV2/tensor_namessave_6/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1v/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1v/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1v/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1v/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1v/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1v/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1*6
dtypes,
*2(
Щ
save_6/control_dependencyIdentitysave_6/ShardedFilename^save_6/SaveV2*
_output_shapes
: *
T0*)
_class
loc:@save_6/ShardedFilename
£
-save_6/MergeV2Checkpoints/checkpoint_prefixesPacksave_6/ShardedFilename^save_6/control_dependency*
T0*
N*
_output_shapes
:*

axis 
Г
save_6/MergeV2CheckpointsMergeV2Checkpoints-save_6/MergeV2Checkpoints/checkpoint_prefixessave_6/Const*
delete_old_dirs(
В
save_6/IdentityIdentitysave_6/Const^save_6/MergeV2Checkpoints^save_6/control_dependency*
_output_shapes
: *
T0
Н
save_6/RestoreV2/tensor_namesConst*
_output_shapes
:(*ї
value±BЃ(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
dtype0
Є
!save_6/RestoreV2/shape_and_slicesConst*
_output_shapes
:(*
dtype0*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
ё
save_6/RestoreV2	RestoreV2save_6/Constsave_6/RestoreV2/tensor_names!save_6/RestoreV2/shape_and_slices*ґ
_output_shapes£
†::::::::::::::::::::::::::::::::::::::::*6
dtypes,
*2(
Ґ
save_6/AssignAssignbeta1_powersave_6/RestoreV2*
_output_shapes
: *
use_locking(* 
_class
loc:@pi/dense/bias*
validate_shape(*
T0
І
save_6/Assign_1Assignbeta1_power_1save_6/RestoreV2:1*
validate_shape(*
T0*
_class
loc:@v/dense/bias*
use_locking(*
_output_shapes
: 
¶
save_6/Assign_2Assignbeta2_powersave_6/RestoreV2:2*
validate_shape(*
_output_shapes
: *
T0*
use_locking(* 
_class
loc:@pi/dense/bias
І
save_6/Assign_3Assignbeta2_power_1save_6/RestoreV2:3*
_class
loc:@v/dense/bias*
use_locking(*
validate_shape(*
T0*
_output_shapes
: 
ђ
save_6/Assign_4Assignpi/dense/biassave_6/RestoreV2:4*
_output_shapes
:@*
T0*
use_locking(*
validate_shape(* 
_class
loc:@pi/dense/bias
±
save_6/Assign_5Assignpi/dense/bias/Adamsave_6/RestoreV2:5*
validate_shape(*
T0*
use_locking(* 
_class
loc:@pi/dense/bias*
_output_shapes
:@
≥
save_6/Assign_6Assignpi/dense/bias/Adam_1save_6/RestoreV2:6*
T0*
validate_shape(*
use_locking(*
_output_shapes
:@* 
_class
loc:@pi/dense/bias
і
save_6/Assign_7Assignpi/dense/kernelsave_6/RestoreV2:7*
validate_shape(*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel*
_output_shapes

:@
є
save_6/Assign_8Assignpi/dense/kernel/Adamsave_6/RestoreV2:8*
use_locking(*
T0*
_output_shapes

:@*
validate_shape(*"
_class
loc:@pi/dense/kernel
ї
save_6/Assign_9Assignpi/dense/kernel/Adam_1save_6/RestoreV2:9*"
_class
loc:@pi/dense/kernel*
use_locking(*
T0*
_output_shapes

:@*
validate_shape(
≤
save_6/Assign_10Assignpi/dense_1/biassave_6/RestoreV2:10*
validate_shape(*"
_class
loc:@pi/dense_1/bias*
T0*
_output_shapes
:@*
use_locking(
Ј
save_6/Assign_11Assignpi/dense_1/bias/Adamsave_6/RestoreV2:11*
use_locking(*
validate_shape(*
T0*
_output_shapes
:@*"
_class
loc:@pi/dense_1/bias
є
save_6/Assign_12Assignpi/dense_1/bias/Adam_1save_6/RestoreV2:12*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
use_locking(*
T0*
_output_shapes
:@
Ї
save_6/Assign_13Assignpi/dense_1/kernelsave_6/RestoreV2:13*
T0*
use_locking(*$
_class
loc:@pi/dense_1/kernel*
_output_shapes

:@@*
validate_shape(
њ
save_6/Assign_14Assignpi/dense_1/kernel/Adamsave_6/RestoreV2:14*$
_class
loc:@pi/dense_1/kernel*
validate_shape(*
T0*
_output_shapes

:@@*
use_locking(
Ѕ
save_6/Assign_15Assignpi/dense_1/kernel/Adam_1save_6/RestoreV2:15*
use_locking(*$
_class
loc:@pi/dense_1/kernel*
_output_shapes

:@@*
validate_shape(*
T0
≤
save_6/Assign_16Assignpi/dense_2/biassave_6/RestoreV2:16*
use_locking(*
_output_shapes
:*
validate_shape(*"
_class
loc:@pi/dense_2/bias*
T0
Ј
save_6/Assign_17Assignpi/dense_2/bias/Adamsave_6/RestoreV2:17*
T0*
_output_shapes
:*
validate_shape(*"
_class
loc:@pi/dense_2/bias*
use_locking(
є
save_6/Assign_18Assignpi/dense_2/bias/Adam_1save_6/RestoreV2:18*
_output_shapes
:*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
T0*
use_locking(
Ї
save_6/Assign_19Assignpi/dense_2/kernelsave_6/RestoreV2:19*
_output_shapes

:@*
use_locking(*
validate_shape(*
T0*$
_class
loc:@pi/dense_2/kernel
њ
save_6/Assign_20Assignpi/dense_2/kernel/Adamsave_6/RestoreV2:20*$
_class
loc:@pi/dense_2/kernel*
use_locking(*
validate_shape(*
T0*
_output_shapes

:@
Ѕ
save_6/Assign_21Assignpi/dense_2/kernel/Adam_1save_6/RestoreV2:21*
T0*
_output_shapes

:@*
validate_shape(*
use_locking(*$
_class
loc:@pi/dense_2/kernel
ђ
save_6/Assign_22Assignv/dense/biassave_6/RestoreV2:22*
_output_shapes
:@*
T0*
validate_shape(*
use_locking(*
_class
loc:@v/dense/bias
±
save_6/Assign_23Assignv/dense/bias/Adamsave_6/RestoreV2:23*
_class
loc:@v/dense/bias*
_output_shapes
:@*
validate_shape(*
use_locking(*
T0
≥
save_6/Assign_24Assignv/dense/bias/Adam_1save_6/RestoreV2:24*
use_locking(*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
:@
і
save_6/Assign_25Assignv/dense/kernelsave_6/RestoreV2:25*
T0*
use_locking(*!
_class
loc:@v/dense/kernel*
_output_shapes

:@*
validate_shape(
є
save_6/Assign_26Assignv/dense/kernel/Adamsave_6/RestoreV2:26*
T0*
validate_shape(*!
_class
loc:@v/dense/kernel*
_output_shapes

:@*
use_locking(
ї
save_6/Assign_27Assignv/dense/kernel/Adam_1save_6/RestoreV2:27*
T0*
validate_shape(*
use_locking(*!
_class
loc:@v/dense/kernel*
_output_shapes

:@
∞
save_6/Assign_28Assignv/dense_1/biassave_6/RestoreV2:28*
T0*
use_locking(*!
_class
loc:@v/dense_1/bias*
_output_shapes
:@*
validate_shape(
µ
save_6/Assign_29Assignv/dense_1/bias/Adamsave_6/RestoreV2:29*!
_class
loc:@v/dense_1/bias*
use_locking(*
_output_shapes
:@*
T0*
validate_shape(
Ј
save_6/Assign_30Assignv/dense_1/bias/Adam_1save_6/RestoreV2:30*
T0*
use_locking(*
_output_shapes
:@*
validate_shape(*!
_class
loc:@v/dense_1/bias
Є
save_6/Assign_31Assignv/dense_1/kernelsave_6/RestoreV2:31*
T0*
validate_shape(*
_output_shapes

:@@*
use_locking(*#
_class
loc:@v/dense_1/kernel
љ
save_6/Assign_32Assignv/dense_1/kernel/Adamsave_6/RestoreV2:32*#
_class
loc:@v/dense_1/kernel*
T0*
_output_shapes

:@@*
validate_shape(*
use_locking(
њ
save_6/Assign_33Assignv/dense_1/kernel/Adam_1save_6/RestoreV2:33*
_output_shapes

:@@*
validate_shape(*#
_class
loc:@v/dense_1/kernel*
T0*
use_locking(
∞
save_6/Assign_34Assignv/dense_2/biassave_6/RestoreV2:34*
validate_shape(*
_output_shapes
:*
use_locking(*!
_class
loc:@v/dense_2/bias*
T0
µ
save_6/Assign_35Assignv/dense_2/bias/Adamsave_6/RestoreV2:35*
T0*
use_locking(*
_output_shapes
:*!
_class
loc:@v/dense_2/bias*
validate_shape(
Ј
save_6/Assign_36Assignv/dense_2/bias/Adam_1save_6/RestoreV2:36*
validate_shape(*
_output_shapes
:*
T0*!
_class
loc:@v/dense_2/bias*
use_locking(
Є
save_6/Assign_37Assignv/dense_2/kernelsave_6/RestoreV2:37*#
_class
loc:@v/dense_2/kernel*
T0*
validate_shape(*
use_locking(*
_output_shapes

:@
љ
save_6/Assign_38Assignv/dense_2/kernel/Adamsave_6/RestoreV2:38*
T0*
use_locking(*
_output_shapes

:@*
validate_shape(*#
_class
loc:@v/dense_2/kernel
њ
save_6/Assign_39Assignv/dense_2/kernel/Adam_1save_6/RestoreV2:39*
T0*
use_locking(*#
_class
loc:@v/dense_2/kernel*
validate_shape(*
_output_shapes

:@
И
save_6/restore_shardNoOp^save_6/Assign^save_6/Assign_1^save_6/Assign_10^save_6/Assign_11^save_6/Assign_12^save_6/Assign_13^save_6/Assign_14^save_6/Assign_15^save_6/Assign_16^save_6/Assign_17^save_6/Assign_18^save_6/Assign_19^save_6/Assign_2^save_6/Assign_20^save_6/Assign_21^save_6/Assign_22^save_6/Assign_23^save_6/Assign_24^save_6/Assign_25^save_6/Assign_26^save_6/Assign_27^save_6/Assign_28^save_6/Assign_29^save_6/Assign_3^save_6/Assign_30^save_6/Assign_31^save_6/Assign_32^save_6/Assign_33^save_6/Assign_34^save_6/Assign_35^save_6/Assign_36^save_6/Assign_37^save_6/Assign_38^save_6/Assign_39^save_6/Assign_4^save_6/Assign_5^save_6/Assign_6^save_6/Assign_7^save_6/Assign_8^save_6/Assign_9
1
save_6/restore_allNoOp^save_6/restore_shard
[
save_7/filename/inputConst*
_output_shapes
: *
valueB Bmodel*
dtype0
r
save_7/filenamePlaceholderWithDefaultsave_7/filename/input*
_output_shapes
: *
shape: *
dtype0
i
save_7/ConstPlaceholderWithDefaultsave_7/filename*
shape: *
dtype0*
_output_shapes
: 
Ж
save_7/StringJoin/inputs_1Const*
dtype0*<
value3B1 B+_temp_48c357cdb5cc4633a21a43389229d329/part*
_output_shapes
: 
{
save_7/StringJoin
StringJoinsave_7/Constsave_7/StringJoin/inputs_1*
_output_shapes
: *
N*
	separator 
S
save_7/num_shardsConst*
value	B :*
_output_shapes
: *
dtype0
^
save_7/ShardedFilename/shardConst*
value	B : *
_output_shapes
: *
dtype0
Е
save_7/ShardedFilenameShardedFilenamesave_7/StringJoinsave_7/ShardedFilename/shardsave_7/num_shards*
_output_shapes
: 
К
save_7/SaveV2/tensor_namesConst*
dtype0*ї
value±BЃ(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
_output_shapes
:(
µ
save_7/SaveV2/shape_and_slicesConst*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:(*
dtype0
…
save_7/SaveV2SaveV2save_7/ShardedFilenamesave_7/SaveV2/tensor_namessave_7/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1v/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1v/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1v/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1v/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1v/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1v/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1*6
dtypes,
*2(
Щ
save_7/control_dependencyIdentitysave_7/ShardedFilename^save_7/SaveV2*
_output_shapes
: *
T0*)
_class
loc:@save_7/ShardedFilename
£
-save_7/MergeV2Checkpoints/checkpoint_prefixesPacksave_7/ShardedFilename^save_7/control_dependency*

axis *
_output_shapes
:*
T0*
N
Г
save_7/MergeV2CheckpointsMergeV2Checkpoints-save_7/MergeV2Checkpoints/checkpoint_prefixessave_7/Const*
delete_old_dirs(
В
save_7/IdentityIdentitysave_7/Const^save_7/MergeV2Checkpoints^save_7/control_dependency*
T0*
_output_shapes
: 
Н
save_7/RestoreV2/tensor_namesConst*
dtype0*
_output_shapes
:(*ї
value±BЃ(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1
Є
!save_7/RestoreV2/shape_and_slicesConst*
_output_shapes
:(*
dtype0*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
ё
save_7/RestoreV2	RestoreV2save_7/Constsave_7/RestoreV2/tensor_names!save_7/RestoreV2/shape_and_slices*6
dtypes,
*2(*ґ
_output_shapes£
†::::::::::::::::::::::::::::::::::::::::
Ґ
save_7/AssignAssignbeta1_powersave_7/RestoreV2*
T0*
_output_shapes
: *
validate_shape(*
use_locking(* 
_class
loc:@pi/dense/bias
І
save_7/Assign_1Assignbeta1_power_1save_7/RestoreV2:1*
T0*
_output_shapes
: *
use_locking(*
validate_shape(*
_class
loc:@v/dense/bias
¶
save_7/Assign_2Assignbeta2_powersave_7/RestoreV2:2*
T0*
use_locking(*
_output_shapes
: * 
_class
loc:@pi/dense/bias*
validate_shape(
І
save_7/Assign_3Assignbeta2_power_1save_7/RestoreV2:3*
T0*
_output_shapes
: *
_class
loc:@v/dense/bias*
use_locking(*
validate_shape(
ђ
save_7/Assign_4Assignpi/dense/biassave_7/RestoreV2:4* 
_class
loc:@pi/dense/bias*
T0*
use_locking(*
validate_shape(*
_output_shapes
:@
±
save_7/Assign_5Assignpi/dense/bias/Adamsave_7/RestoreV2:5*
validate_shape(*
T0* 
_class
loc:@pi/dense/bias*
_output_shapes
:@*
use_locking(
≥
save_7/Assign_6Assignpi/dense/bias/Adam_1save_7/RestoreV2:6*
T0* 
_class
loc:@pi/dense/bias*
use_locking(*
validate_shape(*
_output_shapes
:@
і
save_7/Assign_7Assignpi/dense/kernelsave_7/RestoreV2:7*"
_class
loc:@pi/dense/kernel*
validate_shape(*
T0*
_output_shapes

:@*
use_locking(
є
save_7/Assign_8Assignpi/dense/kernel/Adamsave_7/RestoreV2:8*"
_class
loc:@pi/dense/kernel*
T0*
_output_shapes

:@*
use_locking(*
validate_shape(
ї
save_7/Assign_9Assignpi/dense/kernel/Adam_1save_7/RestoreV2:9*
_output_shapes

:@*"
_class
loc:@pi/dense/kernel*
validate_shape(*
use_locking(*
T0
≤
save_7/Assign_10Assignpi/dense_1/biassave_7/RestoreV2:10*"
_class
loc:@pi/dense_1/bias*
T0*
_output_shapes
:@*
validate_shape(*
use_locking(
Ј
save_7/Assign_11Assignpi/dense_1/bias/Adamsave_7/RestoreV2:11*"
_class
loc:@pi/dense_1/bias*
_output_shapes
:@*
validate_shape(*
use_locking(*
T0
є
save_7/Assign_12Assignpi/dense_1/bias/Adam_1save_7/RestoreV2:12*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
use_locking(*
T0*
_output_shapes
:@
Ї
save_7/Assign_13Assignpi/dense_1/kernelsave_7/RestoreV2:13*
_output_shapes

:@@*$
_class
loc:@pi/dense_1/kernel*
validate_shape(*
use_locking(*
T0
њ
save_7/Assign_14Assignpi/dense_1/kernel/Adamsave_7/RestoreV2:14*
use_locking(*$
_class
loc:@pi/dense_1/kernel*
_output_shapes

:@@*
validate_shape(*
T0
Ѕ
save_7/Assign_15Assignpi/dense_1/kernel/Adam_1save_7/RestoreV2:15*
T0*
_output_shapes

:@@*
validate_shape(*$
_class
loc:@pi/dense_1/kernel*
use_locking(
≤
save_7/Assign_16Assignpi/dense_2/biassave_7/RestoreV2:16*
validate_shape(*
T0*
use_locking(*
_output_shapes
:*"
_class
loc:@pi/dense_2/bias
Ј
save_7/Assign_17Assignpi/dense_2/bias/Adamsave_7/RestoreV2:17*
_output_shapes
:*
T0*
use_locking(*
validate_shape(*"
_class
loc:@pi/dense_2/bias
є
save_7/Assign_18Assignpi/dense_2/bias/Adam_1save_7/RestoreV2:18*
validate_shape(*
use_locking(*
_output_shapes
:*
T0*"
_class
loc:@pi/dense_2/bias
Ї
save_7/Assign_19Assignpi/dense_2/kernelsave_7/RestoreV2:19*
validate_shape(*$
_class
loc:@pi/dense_2/kernel*
_output_shapes

:@*
T0*
use_locking(
њ
save_7/Assign_20Assignpi/dense_2/kernel/Adamsave_7/RestoreV2:20*
use_locking(*
T0*$
_class
loc:@pi/dense_2/kernel*
_output_shapes

:@*
validate_shape(
Ѕ
save_7/Assign_21Assignpi/dense_2/kernel/Adam_1save_7/RestoreV2:21*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@*
T0*
use_locking(
ђ
save_7/Assign_22Assignv/dense/biassave_7/RestoreV2:22*
_class
loc:@v/dense/bias*
use_locking(*
validate_shape(*
T0*
_output_shapes
:@
±
save_7/Assign_23Assignv/dense/bias/Adamsave_7/RestoreV2:23*
validate_shape(*
_class
loc:@v/dense/bias*
T0*
use_locking(*
_output_shapes
:@
≥
save_7/Assign_24Assignv/dense/bias/Adam_1save_7/RestoreV2:24*
T0*
use_locking(*
validate_shape(*
_class
loc:@v/dense/bias*
_output_shapes
:@
і
save_7/Assign_25Assignv/dense/kernelsave_7/RestoreV2:25*
T0*!
_class
loc:@v/dense/kernel*
validate_shape(*
_output_shapes

:@*
use_locking(
є
save_7/Assign_26Assignv/dense/kernel/Adamsave_7/RestoreV2:26*
use_locking(*
validate_shape(*!
_class
loc:@v/dense/kernel*
_output_shapes

:@*
T0
ї
save_7/Assign_27Assignv/dense/kernel/Adam_1save_7/RestoreV2:27*
validate_shape(*
T0*
use_locking(*!
_class
loc:@v/dense/kernel*
_output_shapes

:@
∞
save_7/Assign_28Assignv/dense_1/biassave_7/RestoreV2:28*
validate_shape(*
use_locking(*
T0*!
_class
loc:@v/dense_1/bias*
_output_shapes
:@
µ
save_7/Assign_29Assignv/dense_1/bias/Adamsave_7/RestoreV2:29*
_output_shapes
:@*!
_class
loc:@v/dense_1/bias*
use_locking(*
validate_shape(*
T0
Ј
save_7/Assign_30Assignv/dense_1/bias/Adam_1save_7/RestoreV2:30*!
_class
loc:@v/dense_1/bias*
T0*
use_locking(*
validate_shape(*
_output_shapes
:@
Є
save_7/Assign_31Assignv/dense_1/kernelsave_7/RestoreV2:31*
use_locking(*
T0*
validate_shape(*#
_class
loc:@v/dense_1/kernel*
_output_shapes

:@@
љ
save_7/Assign_32Assignv/dense_1/kernel/Adamsave_7/RestoreV2:32*
T0*
validate_shape(*
use_locking(*#
_class
loc:@v/dense_1/kernel*
_output_shapes

:@@
њ
save_7/Assign_33Assignv/dense_1/kernel/Adam_1save_7/RestoreV2:33*
validate_shape(*#
_class
loc:@v/dense_1/kernel*
use_locking(*
T0*
_output_shapes

:@@
∞
save_7/Assign_34Assignv/dense_2/biassave_7/RestoreV2:34*
validate_shape(*!
_class
loc:@v/dense_2/bias*
_output_shapes
:*
use_locking(*
T0
µ
save_7/Assign_35Assignv/dense_2/bias/Adamsave_7/RestoreV2:35*
T0*!
_class
loc:@v/dense_2/bias*
_output_shapes
:*
validate_shape(*
use_locking(
Ј
save_7/Assign_36Assignv/dense_2/bias/Adam_1save_7/RestoreV2:36*
T0*
_output_shapes
:*
validate_shape(*
use_locking(*!
_class
loc:@v/dense_2/bias
Є
save_7/Assign_37Assignv/dense_2/kernelsave_7/RestoreV2:37*
T0*
_output_shapes

:@*
validate_shape(*#
_class
loc:@v/dense_2/kernel*
use_locking(
љ
save_7/Assign_38Assignv/dense_2/kernel/Adamsave_7/RestoreV2:38*#
_class
loc:@v/dense_2/kernel*
T0*
_output_shapes

:@*
validate_shape(*
use_locking(
њ
save_7/Assign_39Assignv/dense_2/kernel/Adam_1save_7/RestoreV2:39*
use_locking(*
T0*
_output_shapes

:@*#
_class
loc:@v/dense_2/kernel*
validate_shape(
И
save_7/restore_shardNoOp^save_7/Assign^save_7/Assign_1^save_7/Assign_10^save_7/Assign_11^save_7/Assign_12^save_7/Assign_13^save_7/Assign_14^save_7/Assign_15^save_7/Assign_16^save_7/Assign_17^save_7/Assign_18^save_7/Assign_19^save_7/Assign_2^save_7/Assign_20^save_7/Assign_21^save_7/Assign_22^save_7/Assign_23^save_7/Assign_24^save_7/Assign_25^save_7/Assign_26^save_7/Assign_27^save_7/Assign_28^save_7/Assign_29^save_7/Assign_3^save_7/Assign_30^save_7/Assign_31^save_7/Assign_32^save_7/Assign_33^save_7/Assign_34^save_7/Assign_35^save_7/Assign_36^save_7/Assign_37^save_7/Assign_38^save_7/Assign_39^save_7/Assign_4^save_7/Assign_5^save_7/Assign_6^save_7/Assign_7^save_7/Assign_8^save_7/Assign_9
1
save_7/restore_allNoOp^save_7/restore_shard
[
save_8/filename/inputConst*
valueB Bmodel*
_output_shapes
: *
dtype0
r
save_8/filenamePlaceholderWithDefaultsave_8/filename/input*
shape: *
_output_shapes
: *
dtype0
i
save_8/ConstPlaceholderWithDefaultsave_8/filename*
_output_shapes
: *
dtype0*
shape: 
Ж
save_8/StringJoin/inputs_1Const*<
value3B1 B+_temp_b0e6262e5b9744aeb2ae9f2736850b96/part*
dtype0*
_output_shapes
: 
{
save_8/StringJoin
StringJoinsave_8/Constsave_8/StringJoin/inputs_1*
_output_shapes
: *
	separator *
N
S
save_8/num_shardsConst*
_output_shapes
: *
dtype0*
value	B :
^
save_8/ShardedFilename/shardConst*
dtype0*
_output_shapes
: *
value	B : 
Е
save_8/ShardedFilenameShardedFilenamesave_8/StringJoinsave_8/ShardedFilename/shardsave_8/num_shards*
_output_shapes
: 
К
save_8/SaveV2/tensor_namesConst*
dtype0*ї
value±BЃ(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
_output_shapes
:(
µ
save_8/SaveV2/shape_and_slicesConst*
_output_shapes
:(*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
…
save_8/SaveV2SaveV2save_8/ShardedFilenamesave_8/SaveV2/tensor_namessave_8/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1v/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1v/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1v/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1v/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1v/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1v/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1*6
dtypes,
*2(
Щ
save_8/control_dependencyIdentitysave_8/ShardedFilename^save_8/SaveV2*)
_class
loc:@save_8/ShardedFilename*
_output_shapes
: *
T0
£
-save_8/MergeV2Checkpoints/checkpoint_prefixesPacksave_8/ShardedFilename^save_8/control_dependency*
T0*

axis *
_output_shapes
:*
N
Г
save_8/MergeV2CheckpointsMergeV2Checkpoints-save_8/MergeV2Checkpoints/checkpoint_prefixessave_8/Const*
delete_old_dirs(
В
save_8/IdentityIdentitysave_8/Const^save_8/MergeV2Checkpoints^save_8/control_dependency*
_output_shapes
: *
T0
Н
save_8/RestoreV2/tensor_namesConst*ї
value±BЃ(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:(
Є
!save_8/RestoreV2/shape_and_slicesConst*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:(*
dtype0
ё
save_8/RestoreV2	RestoreV2save_8/Constsave_8/RestoreV2/tensor_names!save_8/RestoreV2/shape_and_slices*ґ
_output_shapes£
†::::::::::::::::::::::::::::::::::::::::*6
dtypes,
*2(
Ґ
save_8/AssignAssignbeta1_powersave_8/RestoreV2* 
_class
loc:@pi/dense/bias*
T0*
_output_shapes
: *
validate_shape(*
use_locking(
І
save_8/Assign_1Assignbeta1_power_1save_8/RestoreV2:1*
_class
loc:@v/dense/bias*
use_locking(*
T0*
validate_shape(*
_output_shapes
: 
¶
save_8/Assign_2Assignbeta2_powersave_8/RestoreV2:2* 
_class
loc:@pi/dense/bias*
use_locking(*
validate_shape(*
_output_shapes
: *
T0
І
save_8/Assign_3Assignbeta2_power_1save_8/RestoreV2:3*
validate_shape(*
_class
loc:@v/dense/bias*
T0*
use_locking(*
_output_shapes
: 
ђ
save_8/Assign_4Assignpi/dense/biassave_8/RestoreV2:4*
_output_shapes
:@*
validate_shape(*
T0*
use_locking(* 
_class
loc:@pi/dense/bias
±
save_8/Assign_5Assignpi/dense/bias/Adamsave_8/RestoreV2:5*
validate_shape(*
T0* 
_class
loc:@pi/dense/bias*
_output_shapes
:@*
use_locking(
≥
save_8/Assign_6Assignpi/dense/bias/Adam_1save_8/RestoreV2:6*
_output_shapes
:@*
use_locking(*
validate_shape(* 
_class
loc:@pi/dense/bias*
T0
і
save_8/Assign_7Assignpi/dense/kernelsave_8/RestoreV2:7*
T0*
_output_shapes

:@*"
_class
loc:@pi/dense/kernel*
use_locking(*
validate_shape(
є
save_8/Assign_8Assignpi/dense/kernel/Adamsave_8/RestoreV2:8*
use_locking(*"
_class
loc:@pi/dense/kernel*
T0*
_output_shapes

:@*
validate_shape(
ї
save_8/Assign_9Assignpi/dense/kernel/Adam_1save_8/RestoreV2:9*"
_class
loc:@pi/dense/kernel*
T0*
validate_shape(*
use_locking(*
_output_shapes

:@
≤
save_8/Assign_10Assignpi/dense_1/biassave_8/RestoreV2:10*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
T0*
_output_shapes
:@*
use_locking(
Ј
save_8/Assign_11Assignpi/dense_1/bias/Adamsave_8/RestoreV2:11*
T0*"
_class
loc:@pi/dense_1/bias*
use_locking(*
_output_shapes
:@*
validate_shape(
є
save_8/Assign_12Assignpi/dense_1/bias/Adam_1save_8/RestoreV2:12*
validate_shape(*
_output_shapes
:@*
use_locking(*"
_class
loc:@pi/dense_1/bias*
T0
Ї
save_8/Assign_13Assignpi/dense_1/kernelsave_8/RestoreV2:13*
validate_shape(*$
_class
loc:@pi/dense_1/kernel*
_output_shapes

:@@*
use_locking(*
T0
њ
save_8/Assign_14Assignpi/dense_1/kernel/Adamsave_8/RestoreV2:14*
use_locking(*$
_class
loc:@pi/dense_1/kernel*
_output_shapes

:@@*
validate_shape(*
T0
Ѕ
save_8/Assign_15Assignpi/dense_1/kernel/Adam_1save_8/RestoreV2:15*
T0*$
_class
loc:@pi/dense_1/kernel*
_output_shapes

:@@*
validate_shape(*
use_locking(
≤
save_8/Assign_16Assignpi/dense_2/biassave_8/RestoreV2:16*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
T0*
use_locking(*
_output_shapes
:
Ј
save_8/Assign_17Assignpi/dense_2/bias/Adamsave_8/RestoreV2:17*
use_locking(*
validate_shape(*
T0*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:
є
save_8/Assign_18Assignpi/dense_2/bias/Adam_1save_8/RestoreV2:18*
use_locking(*
T0*
_output_shapes
:*
validate_shape(*"
_class
loc:@pi/dense_2/bias
Ї
save_8/Assign_19Assignpi/dense_2/kernelsave_8/RestoreV2:19*
T0*
use_locking(*
validate_shape(*
_output_shapes

:@*$
_class
loc:@pi/dense_2/kernel
њ
save_8/Assign_20Assignpi/dense_2/kernel/Adamsave_8/RestoreV2:20*
use_locking(*
validate_shape(*$
_class
loc:@pi/dense_2/kernel*
T0*
_output_shapes

:@
Ѕ
save_8/Assign_21Assignpi/dense_2/kernel/Adam_1save_8/RestoreV2:21*
T0*
use_locking(*$
_class
loc:@pi/dense_2/kernel*
_output_shapes

:@*
validate_shape(
ђ
save_8/Assign_22Assignv/dense/biassave_8/RestoreV2:22*
T0*
_output_shapes
:@*
use_locking(*
validate_shape(*
_class
loc:@v/dense/bias
±
save_8/Assign_23Assignv/dense/bias/Adamsave_8/RestoreV2:23*
use_locking(*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
:@*
T0
≥
save_8/Assign_24Assignv/dense/bias/Adam_1save_8/RestoreV2:24*
_class
loc:@v/dense/bias*
T0*
_output_shapes
:@*
validate_shape(*
use_locking(
і
save_8/Assign_25Assignv/dense/kernelsave_8/RestoreV2:25*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0*!
_class
loc:@v/dense/kernel
є
save_8/Assign_26Assignv/dense/kernel/Adamsave_8/RestoreV2:26*
_output_shapes

:@*!
_class
loc:@v/dense/kernel*
use_locking(*
validate_shape(*
T0
ї
save_8/Assign_27Assignv/dense/kernel/Adam_1save_8/RestoreV2:27*
use_locking(*
_output_shapes

:@*
T0*
validate_shape(*!
_class
loc:@v/dense/kernel
∞
save_8/Assign_28Assignv/dense_1/biassave_8/RestoreV2:28*
validate_shape(*
use_locking(*
_output_shapes
:@*!
_class
loc:@v/dense_1/bias*
T0
µ
save_8/Assign_29Assignv/dense_1/bias/Adamsave_8/RestoreV2:29*
validate_shape(*!
_class
loc:@v/dense_1/bias*
_output_shapes
:@*
use_locking(*
T0
Ј
save_8/Assign_30Assignv/dense_1/bias/Adam_1save_8/RestoreV2:30*!
_class
loc:@v/dense_1/bias*
validate_shape(*
use_locking(*
T0*
_output_shapes
:@
Є
save_8/Assign_31Assignv/dense_1/kernelsave_8/RestoreV2:31*
use_locking(*#
_class
loc:@v/dense_1/kernel*
validate_shape(*
T0*
_output_shapes

:@@
љ
save_8/Assign_32Assignv/dense_1/kernel/Adamsave_8/RestoreV2:32*
_output_shapes

:@@*
use_locking(*
validate_shape(*#
_class
loc:@v/dense_1/kernel*
T0
њ
save_8/Assign_33Assignv/dense_1/kernel/Adam_1save_8/RestoreV2:33*#
_class
loc:@v/dense_1/kernel*
_output_shapes

:@@*
use_locking(*
T0*
validate_shape(
∞
save_8/Assign_34Assignv/dense_2/biassave_8/RestoreV2:34*
T0*
use_locking(*
validate_shape(*
_output_shapes
:*!
_class
loc:@v/dense_2/bias
µ
save_8/Assign_35Assignv/dense_2/bias/Adamsave_8/RestoreV2:35*!
_class
loc:@v/dense_2/bias*
validate_shape(*
use_locking(*
T0*
_output_shapes
:
Ј
save_8/Assign_36Assignv/dense_2/bias/Adam_1save_8/RestoreV2:36*!
_class
loc:@v/dense_2/bias*
_output_shapes
:*
T0*
use_locking(*
validate_shape(
Є
save_8/Assign_37Assignv/dense_2/kernelsave_8/RestoreV2:37*#
_class
loc:@v/dense_2/kernel*
T0*
_output_shapes

:@*
use_locking(*
validate_shape(
љ
save_8/Assign_38Assignv/dense_2/kernel/Adamsave_8/RestoreV2:38*#
_class
loc:@v/dense_2/kernel*
T0*
_output_shapes

:@*
validate_shape(*
use_locking(
њ
save_8/Assign_39Assignv/dense_2/kernel/Adam_1save_8/RestoreV2:39*
T0*
validate_shape(*#
_class
loc:@v/dense_2/kernel*
use_locking(*
_output_shapes

:@
И
save_8/restore_shardNoOp^save_8/Assign^save_8/Assign_1^save_8/Assign_10^save_8/Assign_11^save_8/Assign_12^save_8/Assign_13^save_8/Assign_14^save_8/Assign_15^save_8/Assign_16^save_8/Assign_17^save_8/Assign_18^save_8/Assign_19^save_8/Assign_2^save_8/Assign_20^save_8/Assign_21^save_8/Assign_22^save_8/Assign_23^save_8/Assign_24^save_8/Assign_25^save_8/Assign_26^save_8/Assign_27^save_8/Assign_28^save_8/Assign_29^save_8/Assign_3^save_8/Assign_30^save_8/Assign_31^save_8/Assign_32^save_8/Assign_33^save_8/Assign_34^save_8/Assign_35^save_8/Assign_36^save_8/Assign_37^save_8/Assign_38^save_8/Assign_39^save_8/Assign_4^save_8/Assign_5^save_8/Assign_6^save_8/Assign_7^save_8/Assign_8^save_8/Assign_9
1
save_8/restore_allNoOp^save_8/restore_shard
[
save_9/filename/inputConst*
dtype0*
_output_shapes
: *
valueB Bmodel
r
save_9/filenamePlaceholderWithDefaultsave_9/filename/input*
dtype0*
_output_shapes
: *
shape: 
i
save_9/ConstPlaceholderWithDefaultsave_9/filename*
_output_shapes
: *
shape: *
dtype0
Ж
save_9/StringJoin/inputs_1Const*<
value3B1 B+_temp_2f2983d4c5e548bf9824a0752db77082/part*
dtype0*
_output_shapes
: 
{
save_9/StringJoin
StringJoinsave_9/Constsave_9/StringJoin/inputs_1*
	separator *
N*
_output_shapes
: 
S
save_9/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
^
save_9/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 
Е
save_9/ShardedFilenameShardedFilenamesave_9/StringJoinsave_9/ShardedFilename/shardsave_9/num_shards*
_output_shapes
: 
К
save_9/SaveV2/tensor_namesConst*
_output_shapes
:(*ї
value±BЃ(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
dtype0
µ
save_9/SaveV2/shape_and_slicesConst*
_output_shapes
:(*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
…
save_9/SaveV2SaveV2save_9/ShardedFilenamesave_9/SaveV2/tensor_namessave_9/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1v/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1v/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1v/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1v/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1v/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1v/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1*6
dtypes,
*2(
Щ
save_9/control_dependencyIdentitysave_9/ShardedFilename^save_9/SaveV2*)
_class
loc:@save_9/ShardedFilename*
_output_shapes
: *
T0
£
-save_9/MergeV2Checkpoints/checkpoint_prefixesPacksave_9/ShardedFilename^save_9/control_dependency*
N*
_output_shapes
:*
T0*

axis 
Г
save_9/MergeV2CheckpointsMergeV2Checkpoints-save_9/MergeV2Checkpoints/checkpoint_prefixessave_9/Const*
delete_old_dirs(
В
save_9/IdentityIdentitysave_9/Const^save_9/MergeV2Checkpoints^save_9/control_dependency*
T0*
_output_shapes
: 
Н
save_9/RestoreV2/tensor_namesConst*ї
value±BЃ(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:(
Є
!save_9/RestoreV2/shape_and_slicesConst*
dtype0*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:(
ё
save_9/RestoreV2	RestoreV2save_9/Constsave_9/RestoreV2/tensor_names!save_9/RestoreV2/shape_and_slices*6
dtypes,
*2(*ґ
_output_shapes£
†::::::::::::::::::::::::::::::::::::::::
Ґ
save_9/AssignAssignbeta1_powersave_9/RestoreV2*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
: 
І
save_9/Assign_1Assignbeta1_power_1save_9/RestoreV2:1*
_class
loc:@v/dense/bias*
validate_shape(*
use_locking(*
T0*
_output_shapes
: 
¶
save_9/Assign_2Assignbeta2_powersave_9/RestoreV2:2*
validate_shape(* 
_class
loc:@pi/dense/bias*
_output_shapes
: *
T0*
use_locking(
І
save_9/Assign_3Assignbeta2_power_1save_9/RestoreV2:3*
_class
loc:@v/dense/bias*
T0*
validate_shape(*
_output_shapes
: *
use_locking(
ђ
save_9/Assign_4Assignpi/dense/biassave_9/RestoreV2:4*
_output_shapes
:@*
use_locking(*
T0*
validate_shape(* 
_class
loc:@pi/dense/bias
±
save_9/Assign_5Assignpi/dense/bias/Adamsave_9/RestoreV2:5*
use_locking(*
_output_shapes
:@* 
_class
loc:@pi/dense/bias*
validate_shape(*
T0
≥
save_9/Assign_6Assignpi/dense/bias/Adam_1save_9/RestoreV2:6*
use_locking(* 
_class
loc:@pi/dense/bias*
_output_shapes
:@*
validate_shape(*
T0
і
save_9/Assign_7Assignpi/dense/kernelsave_9/RestoreV2:7*
validate_shape(*
T0*"
_class
loc:@pi/dense/kernel*
_output_shapes

:@*
use_locking(
є
save_9/Assign_8Assignpi/dense/kernel/Adamsave_9/RestoreV2:8*"
_class
loc:@pi/dense/kernel*
validate_shape(*
use_locking(*
T0*
_output_shapes

:@
ї
save_9/Assign_9Assignpi/dense/kernel/Adam_1save_9/RestoreV2:9*
validate_shape(*
_output_shapes

:@*"
_class
loc:@pi/dense/kernel*
use_locking(*
T0
≤
save_9/Assign_10Assignpi/dense_1/biassave_9/RestoreV2:10*
T0*"
_class
loc:@pi/dense_1/bias*
_output_shapes
:@*
use_locking(*
validate_shape(
Ј
save_9/Assign_11Assignpi/dense_1/bias/Adamsave_9/RestoreV2:11*"
_class
loc:@pi/dense_1/bias*
_output_shapes
:@*
use_locking(*
T0*
validate_shape(
є
save_9/Assign_12Assignpi/dense_1/bias/Adam_1save_9/RestoreV2:12*
use_locking(*
validate_shape(*
T0*
_output_shapes
:@*"
_class
loc:@pi/dense_1/bias
Ї
save_9/Assign_13Assignpi/dense_1/kernelsave_9/RestoreV2:13*
T0*
use_locking(*
_output_shapes

:@@*$
_class
loc:@pi/dense_1/kernel*
validate_shape(
њ
save_9/Assign_14Assignpi/dense_1/kernel/Adamsave_9/RestoreV2:14*
T0*$
_class
loc:@pi/dense_1/kernel*
use_locking(*
validate_shape(*
_output_shapes

:@@
Ѕ
save_9/Assign_15Assignpi/dense_1/kernel/Adam_1save_9/RestoreV2:15*$
_class
loc:@pi/dense_1/kernel*
use_locking(*
_output_shapes

:@@*
T0*
validate_shape(
≤
save_9/Assign_16Assignpi/dense_2/biassave_9/RestoreV2:16*
use_locking(*
_output_shapes
:*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
T0
Ј
save_9/Assign_17Assignpi/dense_2/bias/Adamsave_9/RestoreV2:17*
validate_shape(*
use_locking(*
_output_shapes
:*"
_class
loc:@pi/dense_2/bias*
T0
є
save_9/Assign_18Assignpi/dense_2/bias/Adam_1save_9/RestoreV2:18*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
use_locking(*
T0*
_output_shapes
:
Ї
save_9/Assign_19Assignpi/dense_2/kernelsave_9/RestoreV2:19*$
_class
loc:@pi/dense_2/kernel*
_output_shapes

:@*
validate_shape(*
use_locking(*
T0
њ
save_9/Assign_20Assignpi/dense_2/kernel/Adamsave_9/RestoreV2:20*
T0*$
_class
loc:@pi/dense_2/kernel*
_output_shapes

:@*
use_locking(*
validate_shape(
Ѕ
save_9/Assign_21Assignpi/dense_2/kernel/Adam_1save_9/RestoreV2:21*
validate_shape(*
_output_shapes

:@*
use_locking(*$
_class
loc:@pi/dense_2/kernel*
T0
ђ
save_9/Assign_22Assignv/dense/biassave_9/RestoreV2:22*
_class
loc:@v/dense/bias*
use_locking(*
validate_shape(*
T0*
_output_shapes
:@
±
save_9/Assign_23Assignv/dense/bias/Adamsave_9/RestoreV2:23*
use_locking(*
T0*
_output_shapes
:@*
_class
loc:@v/dense/bias*
validate_shape(
≥
save_9/Assign_24Assignv/dense/bias/Adam_1save_9/RestoreV2:24*
validate_shape(*
use_locking(*
T0*
_class
loc:@v/dense/bias*
_output_shapes
:@
і
save_9/Assign_25Assignv/dense/kernelsave_9/RestoreV2:25*!
_class
loc:@v/dense/kernel*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0
є
save_9/Assign_26Assignv/dense/kernel/Adamsave_9/RestoreV2:26*
T0*
validate_shape(*
use_locking(*
_output_shapes

:@*!
_class
loc:@v/dense/kernel
ї
save_9/Assign_27Assignv/dense/kernel/Adam_1save_9/RestoreV2:27*
validate_shape(*!
_class
loc:@v/dense/kernel*
T0*
use_locking(*
_output_shapes

:@
∞
save_9/Assign_28Assignv/dense_1/biassave_9/RestoreV2:28*
_output_shapes
:@*
T0*!
_class
loc:@v/dense_1/bias*
use_locking(*
validate_shape(
µ
save_9/Assign_29Assignv/dense_1/bias/Adamsave_9/RestoreV2:29*
_output_shapes
:@*
validate_shape(*
T0*
use_locking(*!
_class
loc:@v/dense_1/bias
Ј
save_9/Assign_30Assignv/dense_1/bias/Adam_1save_9/RestoreV2:30*!
_class
loc:@v/dense_1/bias*
T0*
validate_shape(*
_output_shapes
:@*
use_locking(
Є
save_9/Assign_31Assignv/dense_1/kernelsave_9/RestoreV2:31*
validate_shape(*
_output_shapes

:@@*
T0*
use_locking(*#
_class
loc:@v/dense_1/kernel
љ
save_9/Assign_32Assignv/dense_1/kernel/Adamsave_9/RestoreV2:32*
use_locking(*
_output_shapes

:@@*
validate_shape(*#
_class
loc:@v/dense_1/kernel*
T0
њ
save_9/Assign_33Assignv/dense_1/kernel/Adam_1save_9/RestoreV2:33*
use_locking(*
validate_shape(*
_output_shapes

:@@*
T0*#
_class
loc:@v/dense_1/kernel
∞
save_9/Assign_34Assignv/dense_2/biassave_9/RestoreV2:34*
use_locking(*
_output_shapes
:*
validate_shape(*!
_class
loc:@v/dense_2/bias*
T0
µ
save_9/Assign_35Assignv/dense_2/bias/Adamsave_9/RestoreV2:35*
_output_shapes
:*
T0*!
_class
loc:@v/dense_2/bias*
use_locking(*
validate_shape(
Ј
save_9/Assign_36Assignv/dense_2/bias/Adam_1save_9/RestoreV2:36*!
_class
loc:@v/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
Є
save_9/Assign_37Assignv/dense_2/kernelsave_9/RestoreV2:37*
validate_shape(*
use_locking(*#
_class
loc:@v/dense_2/kernel*
_output_shapes

:@*
T0
љ
save_9/Assign_38Assignv/dense_2/kernel/Adamsave_9/RestoreV2:38*
use_locking(*
T0*
validate_shape(*
_output_shapes

:@*#
_class
loc:@v/dense_2/kernel
њ
save_9/Assign_39Assignv/dense_2/kernel/Adam_1save_9/RestoreV2:39*
use_locking(*#
_class
loc:@v/dense_2/kernel*
validate_shape(*
_output_shapes

:@*
T0
И
save_9/restore_shardNoOp^save_9/Assign^save_9/Assign_1^save_9/Assign_10^save_9/Assign_11^save_9/Assign_12^save_9/Assign_13^save_9/Assign_14^save_9/Assign_15^save_9/Assign_16^save_9/Assign_17^save_9/Assign_18^save_9/Assign_19^save_9/Assign_2^save_9/Assign_20^save_9/Assign_21^save_9/Assign_22^save_9/Assign_23^save_9/Assign_24^save_9/Assign_25^save_9/Assign_26^save_9/Assign_27^save_9/Assign_28^save_9/Assign_29^save_9/Assign_3^save_9/Assign_30^save_9/Assign_31^save_9/Assign_32^save_9/Assign_33^save_9/Assign_34^save_9/Assign_35^save_9/Assign_36^save_9/Assign_37^save_9/Assign_38^save_9/Assign_39^save_9/Assign_4^save_9/Assign_5^save_9/Assign_6^save_9/Assign_7^save_9/Assign_8^save_9/Assign_9
1
save_9/restore_allNoOp^save_9/restore_shard
\
save_10/filename/inputConst*
valueB Bmodel*
_output_shapes
: *
dtype0
t
save_10/filenamePlaceholderWithDefaultsave_10/filename/input*
shape: *
dtype0*
_output_shapes
: 
k
save_10/ConstPlaceholderWithDefaultsave_10/filename*
_output_shapes
: *
dtype0*
shape: 
З
save_10/StringJoin/inputs_1Const*<
value3B1 B+_temp_26b31beaef7a4265a35a5220dae63186/part*
_output_shapes
: *
dtype0
~
save_10/StringJoin
StringJoinsave_10/Constsave_10/StringJoin/inputs_1*
	separator *
_output_shapes
: *
N
T
save_10/num_shardsConst*
dtype0*
value	B :*
_output_shapes
: 
_
save_10/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 
Й
save_10/ShardedFilenameShardedFilenamesave_10/StringJoinsave_10/ShardedFilename/shardsave_10/num_shards*
_output_shapes
: 
Л
save_10/SaveV2/tensor_namesConst*
_output_shapes
:(*
dtype0*ї
value±BЃ(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1
ґ
save_10/SaveV2/shape_and_slicesConst*
dtype0*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:(
Ќ
save_10/SaveV2SaveV2save_10/ShardedFilenamesave_10/SaveV2/tensor_namessave_10/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1v/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1v/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1v/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1v/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1v/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1v/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1*6
dtypes,
*2(
Э
save_10/control_dependencyIdentitysave_10/ShardedFilename^save_10/SaveV2**
_class 
loc:@save_10/ShardedFilename*
_output_shapes
: *
T0
¶
.save_10/MergeV2Checkpoints/checkpoint_prefixesPacksave_10/ShardedFilename^save_10/control_dependency*

axis *
_output_shapes
:*
T0*
N
Ж
save_10/MergeV2CheckpointsMergeV2Checkpoints.save_10/MergeV2Checkpoints/checkpoint_prefixessave_10/Const*
delete_old_dirs(
Ж
save_10/IdentityIdentitysave_10/Const^save_10/MergeV2Checkpoints^save_10/control_dependency*
_output_shapes
: *
T0
О
save_10/RestoreV2/tensor_namesConst*ї
value±BЃ(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
_output_shapes
:(*
dtype0
є
"save_10/RestoreV2/shape_and_slicesConst*
dtype0*
_output_shapes
:(*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
в
save_10/RestoreV2	RestoreV2save_10/Constsave_10/RestoreV2/tensor_names"save_10/RestoreV2/shape_and_slices*ґ
_output_shapes£
†::::::::::::::::::::::::::::::::::::::::*6
dtypes,
*2(
§
save_10/AssignAssignbeta1_powersave_10/RestoreV2*
_output_shapes
: *
T0*
use_locking(*
validate_shape(* 
_class
loc:@pi/dense/bias
©
save_10/Assign_1Assignbeta1_power_1save_10/RestoreV2:1*
_output_shapes
: *
T0*
use_locking(*
validate_shape(*
_class
loc:@v/dense/bias
®
save_10/Assign_2Assignbeta2_powersave_10/RestoreV2:2*
_output_shapes
: * 
_class
loc:@pi/dense/bias*
validate_shape(*
T0*
use_locking(
©
save_10/Assign_3Assignbeta2_power_1save_10/RestoreV2:3*
use_locking(*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
: *
T0
Ѓ
save_10/Assign_4Assignpi/dense/biassave_10/RestoreV2:4*
validate_shape(*
_output_shapes
:@*
T0* 
_class
loc:@pi/dense/bias*
use_locking(
≥
save_10/Assign_5Assignpi/dense/bias/Adamsave_10/RestoreV2:5*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0* 
_class
loc:@pi/dense/bias
µ
save_10/Assign_6Assignpi/dense/bias/Adam_1save_10/RestoreV2:6*
_output_shapes
:@* 
_class
loc:@pi/dense/bias*
validate_shape(*
use_locking(*
T0
ґ
save_10/Assign_7Assignpi/dense/kernelsave_10/RestoreV2:7*
T0*
validate_shape(*
use_locking(*"
_class
loc:@pi/dense/kernel*
_output_shapes

:@
ї
save_10/Assign_8Assignpi/dense/kernel/Adamsave_10/RestoreV2:8*
T0*"
_class
loc:@pi/dense/kernel*
use_locking(*
_output_shapes

:@*
validate_shape(
љ
save_10/Assign_9Assignpi/dense/kernel/Adam_1save_10/RestoreV2:9*
T0*
use_locking(*
validate_shape(*"
_class
loc:@pi/dense/kernel*
_output_shapes

:@
і
save_10/Assign_10Assignpi/dense_1/biassave_10/RestoreV2:10*
_output_shapes
:@*
use_locking(*"
_class
loc:@pi/dense_1/bias*
T0*
validate_shape(
є
save_10/Assign_11Assignpi/dense_1/bias/Adamsave_10/RestoreV2:11*"
_class
loc:@pi/dense_1/bias*
_output_shapes
:@*
use_locking(*
validate_shape(*
T0
ї
save_10/Assign_12Assignpi/dense_1/bias/Adam_1save_10/RestoreV2:12*
T0*
use_locking(*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
_output_shapes
:@
Љ
save_10/Assign_13Assignpi/dense_1/kernelsave_10/RestoreV2:13*
use_locking(*
_output_shapes

:@@*
T0*
validate_shape(*$
_class
loc:@pi/dense_1/kernel
Ѕ
save_10/Assign_14Assignpi/dense_1/kernel/Adamsave_10/RestoreV2:14*
validate_shape(*
use_locking(*
_output_shapes

:@@*
T0*$
_class
loc:@pi/dense_1/kernel
√
save_10/Assign_15Assignpi/dense_1/kernel/Adam_1save_10/RestoreV2:15*
use_locking(*
T0*
_output_shapes

:@@*
validate_shape(*$
_class
loc:@pi/dense_1/kernel
і
save_10/Assign_16Assignpi/dense_2/biassave_10/RestoreV2:16*
use_locking(*
validate_shape(*
_output_shapes
:*"
_class
loc:@pi/dense_2/bias*
T0
є
save_10/Assign_17Assignpi/dense_2/bias/Adamsave_10/RestoreV2:17*
_output_shapes
:*
T0*
use_locking(*"
_class
loc:@pi/dense_2/bias*
validate_shape(
ї
save_10/Assign_18Assignpi/dense_2/bias/Adam_1save_10/RestoreV2:18*
validate_shape(*
_output_shapes
:*
use_locking(*"
_class
loc:@pi/dense_2/bias*
T0
Љ
save_10/Assign_19Assignpi/dense_2/kernelsave_10/RestoreV2:19*
validate_shape(*
T0*
_output_shapes

:@*
use_locking(*$
_class
loc:@pi/dense_2/kernel
Ѕ
save_10/Assign_20Assignpi/dense_2/kernel/Adamsave_10/RestoreV2:20*
validate_shape(*
use_locking(*$
_class
loc:@pi/dense_2/kernel*
_output_shapes

:@*
T0
√
save_10/Assign_21Assignpi/dense_2/kernel/Adam_1save_10/RestoreV2:21*
use_locking(*
T0*
validate_shape(*$
_class
loc:@pi/dense_2/kernel*
_output_shapes

:@
Ѓ
save_10/Assign_22Assignv/dense/biassave_10/RestoreV2:22*
use_locking(*
T0*
_class
loc:@v/dense/bias*
_output_shapes
:@*
validate_shape(
≥
save_10/Assign_23Assignv/dense/bias/Adamsave_10/RestoreV2:23*
_class
loc:@v/dense/bias*
T0*
_output_shapes
:@*
validate_shape(*
use_locking(
µ
save_10/Assign_24Assignv/dense/bias/Adam_1save_10/RestoreV2:24*
T0*
validate_shape(*
_class
loc:@v/dense/bias*
_output_shapes
:@*
use_locking(
ґ
save_10/Assign_25Assignv/dense/kernelsave_10/RestoreV2:25*
_output_shapes

:@*!
_class
loc:@v/dense/kernel*
T0*
use_locking(*
validate_shape(
ї
save_10/Assign_26Assignv/dense/kernel/Adamsave_10/RestoreV2:26*!
_class
loc:@v/dense/kernel*
_output_shapes

:@*
T0*
use_locking(*
validate_shape(
љ
save_10/Assign_27Assignv/dense/kernel/Adam_1save_10/RestoreV2:27*
use_locking(*
T0*!
_class
loc:@v/dense/kernel*
_output_shapes

:@*
validate_shape(
≤
save_10/Assign_28Assignv/dense_1/biassave_10/RestoreV2:28*
_output_shapes
:@*!
_class
loc:@v/dense_1/bias*
use_locking(*
validate_shape(*
T0
Ј
save_10/Assign_29Assignv/dense_1/bias/Adamsave_10/RestoreV2:29*
validate_shape(*
T0*!
_class
loc:@v/dense_1/bias*
_output_shapes
:@*
use_locking(
є
save_10/Assign_30Assignv/dense_1/bias/Adam_1save_10/RestoreV2:30*
validate_shape(*
T0*
use_locking(*
_output_shapes
:@*!
_class
loc:@v/dense_1/bias
Ї
save_10/Assign_31Assignv/dense_1/kernelsave_10/RestoreV2:31*#
_class
loc:@v/dense_1/kernel*
_output_shapes

:@@*
T0*
validate_shape(*
use_locking(
њ
save_10/Assign_32Assignv/dense_1/kernel/Adamsave_10/RestoreV2:32*
T0*
_output_shapes

:@@*
use_locking(*
validate_shape(*#
_class
loc:@v/dense_1/kernel
Ѕ
save_10/Assign_33Assignv/dense_1/kernel/Adam_1save_10/RestoreV2:33*
validate_shape(*
use_locking(*
_output_shapes

:@@*#
_class
loc:@v/dense_1/kernel*
T0
≤
save_10/Assign_34Assignv/dense_2/biassave_10/RestoreV2:34*
T0*!
_class
loc:@v/dense_2/bias*
use_locking(*
validate_shape(*
_output_shapes
:
Ј
save_10/Assign_35Assignv/dense_2/bias/Adamsave_10/RestoreV2:35*
use_locking(*
_output_shapes
:*!
_class
loc:@v/dense_2/bias*
validate_shape(*
T0
є
save_10/Assign_36Assignv/dense_2/bias/Adam_1save_10/RestoreV2:36*
use_locking(*!
_class
loc:@v/dense_2/bias*
T0*
validate_shape(*
_output_shapes
:
Ї
save_10/Assign_37Assignv/dense_2/kernelsave_10/RestoreV2:37*
use_locking(*
validate_shape(*#
_class
loc:@v/dense_2/kernel*
T0*
_output_shapes

:@
њ
save_10/Assign_38Assignv/dense_2/kernel/Adamsave_10/RestoreV2:38*
use_locking(*
_output_shapes

:@*#
_class
loc:@v/dense_2/kernel*
validate_shape(*
T0
Ѕ
save_10/Assign_39Assignv/dense_2/kernel/Adam_1save_10/RestoreV2:39*
validate_shape(*
T0*
use_locking(*
_output_shapes

:@*#
_class
loc:@v/dense_2/kernel
±
save_10/restore_shardNoOp^save_10/Assign^save_10/Assign_1^save_10/Assign_10^save_10/Assign_11^save_10/Assign_12^save_10/Assign_13^save_10/Assign_14^save_10/Assign_15^save_10/Assign_16^save_10/Assign_17^save_10/Assign_18^save_10/Assign_19^save_10/Assign_2^save_10/Assign_20^save_10/Assign_21^save_10/Assign_22^save_10/Assign_23^save_10/Assign_24^save_10/Assign_25^save_10/Assign_26^save_10/Assign_27^save_10/Assign_28^save_10/Assign_29^save_10/Assign_3^save_10/Assign_30^save_10/Assign_31^save_10/Assign_32^save_10/Assign_33^save_10/Assign_34^save_10/Assign_35^save_10/Assign_36^save_10/Assign_37^save_10/Assign_38^save_10/Assign_39^save_10/Assign_4^save_10/Assign_5^save_10/Assign_6^save_10/Assign_7^save_10/Assign_8^save_10/Assign_9
3
save_10/restore_allNoOp^save_10/restore_shard
\
save_11/filename/inputConst*
valueB Bmodel*
_output_shapes
: *
dtype0
t
save_11/filenamePlaceholderWithDefaultsave_11/filename/input*
shape: *
_output_shapes
: *
dtype0
k
save_11/ConstPlaceholderWithDefaultsave_11/filename*
dtype0*
_output_shapes
: *
shape: 
З
save_11/StringJoin/inputs_1Const*
_output_shapes
: *<
value3B1 B+_temp_f398be3b2c5f460fa9cdf04f46f02f8a/part*
dtype0
~
save_11/StringJoin
StringJoinsave_11/Constsave_11/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
T
save_11/num_shardsConst*
dtype0*
value	B :*
_output_shapes
: 
_
save_11/ShardedFilename/shardConst*
_output_shapes
: *
dtype0*
value	B : 
Й
save_11/ShardedFilenameShardedFilenamesave_11/StringJoinsave_11/ShardedFilename/shardsave_11/num_shards*
_output_shapes
: 
Л
save_11/SaveV2/tensor_namesConst*ї
value±BЃ(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:(
ґ
save_11/SaveV2/shape_and_slicesConst*
dtype0*
_output_shapes
:(*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
Ќ
save_11/SaveV2SaveV2save_11/ShardedFilenamesave_11/SaveV2/tensor_namessave_11/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1v/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1v/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1v/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1v/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1v/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1v/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1*6
dtypes,
*2(
Э
save_11/control_dependencyIdentitysave_11/ShardedFilename^save_11/SaveV2*
T0**
_class 
loc:@save_11/ShardedFilename*
_output_shapes
: 
¶
.save_11/MergeV2Checkpoints/checkpoint_prefixesPacksave_11/ShardedFilename^save_11/control_dependency*
T0*
N*

axis *
_output_shapes
:
Ж
save_11/MergeV2CheckpointsMergeV2Checkpoints.save_11/MergeV2Checkpoints/checkpoint_prefixessave_11/Const*
delete_old_dirs(
Ж
save_11/IdentityIdentitysave_11/Const^save_11/MergeV2Checkpoints^save_11/control_dependency*
T0*
_output_shapes
: 
О
save_11/RestoreV2/tensor_namesConst*ї
value±BЃ(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:(
є
"save_11/RestoreV2/shape_and_slicesConst*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:(*
dtype0
в
save_11/RestoreV2	RestoreV2save_11/Constsave_11/RestoreV2/tensor_names"save_11/RestoreV2/shape_and_slices*6
dtypes,
*2(*ґ
_output_shapes£
†::::::::::::::::::::::::::::::::::::::::
§
save_11/AssignAssignbeta1_powersave_11/RestoreV2* 
_class
loc:@pi/dense/bias*
T0*
_output_shapes
: *
use_locking(*
validate_shape(
©
save_11/Assign_1Assignbeta1_power_1save_11/RestoreV2:1*
_output_shapes
: *
_class
loc:@v/dense/bias*
use_locking(*
T0*
validate_shape(
®
save_11/Assign_2Assignbeta2_powersave_11/RestoreV2:2* 
_class
loc:@pi/dense/bias*
T0*
_output_shapes
: *
validate_shape(*
use_locking(
©
save_11/Assign_3Assignbeta2_power_1save_11/RestoreV2:3*
_output_shapes
: *
use_locking(*
_class
loc:@v/dense/bias*
T0*
validate_shape(
Ѓ
save_11/Assign_4Assignpi/dense/biassave_11/RestoreV2:4*
_output_shapes
:@*
T0*
use_locking(* 
_class
loc:@pi/dense/bias*
validate_shape(
≥
save_11/Assign_5Assignpi/dense/bias/Adamsave_11/RestoreV2:5* 
_class
loc:@pi/dense/bias*
use_locking(*
_output_shapes
:@*
validate_shape(*
T0
µ
save_11/Assign_6Assignpi/dense/bias/Adam_1save_11/RestoreV2:6*
_output_shapes
:@*
validate_shape(*
use_locking(*
T0* 
_class
loc:@pi/dense/bias
ґ
save_11/Assign_7Assignpi/dense/kernelsave_11/RestoreV2:7*
use_locking(*
_output_shapes

:@*
validate_shape(*
T0*"
_class
loc:@pi/dense/kernel
ї
save_11/Assign_8Assignpi/dense/kernel/Adamsave_11/RestoreV2:8*
validate_shape(*
use_locking(*
T0*
_output_shapes

:@*"
_class
loc:@pi/dense/kernel
љ
save_11/Assign_9Assignpi/dense/kernel/Adam_1save_11/RestoreV2:9*
T0*
_output_shapes

:@*
use_locking(*"
_class
loc:@pi/dense/kernel*
validate_shape(
і
save_11/Assign_10Assignpi/dense_1/biassave_11/RestoreV2:10*"
_class
loc:@pi/dense_1/bias*
_output_shapes
:@*
validate_shape(*
T0*
use_locking(
є
save_11/Assign_11Assignpi/dense_1/bias/Adamsave_11/RestoreV2:11*
use_locking(*"
_class
loc:@pi/dense_1/bias*
_output_shapes
:@*
validate_shape(*
T0
ї
save_11/Assign_12Assignpi/dense_1/bias/Adam_1save_11/RestoreV2:12*
validate_shape(*
_output_shapes
:@*"
_class
loc:@pi/dense_1/bias*
use_locking(*
T0
Љ
save_11/Assign_13Assignpi/dense_1/kernelsave_11/RestoreV2:13*$
_class
loc:@pi/dense_1/kernel*
T0*
_output_shapes

:@@*
validate_shape(*
use_locking(
Ѕ
save_11/Assign_14Assignpi/dense_1/kernel/Adamsave_11/RestoreV2:14*
validate_shape(*$
_class
loc:@pi/dense_1/kernel*
use_locking(*
_output_shapes

:@@*
T0
√
save_11/Assign_15Assignpi/dense_1/kernel/Adam_1save_11/RestoreV2:15*$
_class
loc:@pi/dense_1/kernel*
_output_shapes

:@@*
use_locking(*
T0*
validate_shape(
і
save_11/Assign_16Assignpi/dense_2/biassave_11/RestoreV2:16*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
_output_shapes
:*
T0*
use_locking(
є
save_11/Assign_17Assignpi/dense_2/bias/Adamsave_11/RestoreV2:17*
_output_shapes
:*
use_locking(*
validate_shape(*
T0*"
_class
loc:@pi/dense_2/bias
ї
save_11/Assign_18Assignpi/dense_2/bias/Adam_1save_11/RestoreV2:18*
use_locking(*"
_class
loc:@pi/dense_2/bias*
T0*
_output_shapes
:*
validate_shape(
Љ
save_11/Assign_19Assignpi/dense_2/kernelsave_11/RestoreV2:19*
T0*
validate_shape(*
use_locking(*
_output_shapes

:@*$
_class
loc:@pi/dense_2/kernel
Ѕ
save_11/Assign_20Assignpi/dense_2/kernel/Adamsave_11/RestoreV2:20*
validate_shape(*$
_class
loc:@pi/dense_2/kernel*
use_locking(*
T0*
_output_shapes

:@
√
save_11/Assign_21Assignpi/dense_2/kernel/Adam_1save_11/RestoreV2:21*
use_locking(*
T0*
_output_shapes

:@*
validate_shape(*$
_class
loc:@pi/dense_2/kernel
Ѓ
save_11/Assign_22Assignv/dense/biassave_11/RestoreV2:22*
_class
loc:@v/dense/bias*
use_locking(*
T0*
_output_shapes
:@*
validate_shape(
≥
save_11/Assign_23Assignv/dense/bias/Adamsave_11/RestoreV2:23*
use_locking(*
validate_shape(*
T0*
_output_shapes
:@*
_class
loc:@v/dense/bias
µ
save_11/Assign_24Assignv/dense/bias/Adam_1save_11/RestoreV2:24*
_output_shapes
:@*
T0*
validate_shape(*
_class
loc:@v/dense/bias*
use_locking(
ґ
save_11/Assign_25Assignv/dense/kernelsave_11/RestoreV2:25*
use_locking(*
_output_shapes

:@*!
_class
loc:@v/dense/kernel*
validate_shape(*
T0
ї
save_11/Assign_26Assignv/dense/kernel/Adamsave_11/RestoreV2:26*
use_locking(*!
_class
loc:@v/dense/kernel*
T0*
_output_shapes

:@*
validate_shape(
љ
save_11/Assign_27Assignv/dense/kernel/Adam_1save_11/RestoreV2:27*
validate_shape(*
_output_shapes

:@*
T0*!
_class
loc:@v/dense/kernel*
use_locking(
≤
save_11/Assign_28Assignv/dense_1/biassave_11/RestoreV2:28*
validate_shape(*
_output_shapes
:@*
T0*
use_locking(*!
_class
loc:@v/dense_1/bias
Ј
save_11/Assign_29Assignv/dense_1/bias/Adamsave_11/RestoreV2:29*
_output_shapes
:@*
use_locking(*
validate_shape(*!
_class
loc:@v/dense_1/bias*
T0
є
save_11/Assign_30Assignv/dense_1/bias/Adam_1save_11/RestoreV2:30*!
_class
loc:@v/dense_1/bias*
_output_shapes
:@*
validate_shape(*
use_locking(*
T0
Ї
save_11/Assign_31Assignv/dense_1/kernelsave_11/RestoreV2:31*
_output_shapes

:@@*
use_locking(*
T0*
validate_shape(*#
_class
loc:@v/dense_1/kernel
њ
save_11/Assign_32Assignv/dense_1/kernel/Adamsave_11/RestoreV2:32*
validate_shape(*
T0*#
_class
loc:@v/dense_1/kernel*
_output_shapes

:@@*
use_locking(
Ѕ
save_11/Assign_33Assignv/dense_1/kernel/Adam_1save_11/RestoreV2:33*
use_locking(*#
_class
loc:@v/dense_1/kernel*
validate_shape(*
_output_shapes

:@@*
T0
≤
save_11/Assign_34Assignv/dense_2/biassave_11/RestoreV2:34*
_output_shapes
:*
use_locking(*!
_class
loc:@v/dense_2/bias*
validate_shape(*
T0
Ј
save_11/Assign_35Assignv/dense_2/bias/Adamsave_11/RestoreV2:35*
T0*
_output_shapes
:*!
_class
loc:@v/dense_2/bias*
validate_shape(*
use_locking(
є
save_11/Assign_36Assignv/dense_2/bias/Adam_1save_11/RestoreV2:36*
use_locking(*
validate_shape(*
T0*!
_class
loc:@v/dense_2/bias*
_output_shapes
:
Ї
save_11/Assign_37Assignv/dense_2/kernelsave_11/RestoreV2:37*
use_locking(*
validate_shape(*
_output_shapes

:@*#
_class
loc:@v/dense_2/kernel*
T0
њ
save_11/Assign_38Assignv/dense_2/kernel/Adamsave_11/RestoreV2:38*#
_class
loc:@v/dense_2/kernel*
use_locking(*
_output_shapes

:@*
validate_shape(*
T0
Ѕ
save_11/Assign_39Assignv/dense_2/kernel/Adam_1save_11/RestoreV2:39*
use_locking(*
T0*
validate_shape(*
_output_shapes

:@*#
_class
loc:@v/dense_2/kernel
±
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
save_12/filenamePlaceholderWithDefaultsave_12/filename/input*
dtype0*
shape: *
_output_shapes
: 
k
save_12/ConstPlaceholderWithDefaultsave_12/filename*
shape: *
dtype0*
_output_shapes
: 
З
save_12/StringJoin/inputs_1Const*
_output_shapes
: *<
value3B1 B+_temp_e8cb5d67b3ab470d9a72190cc51db16b/part*
dtype0
~
save_12/StringJoin
StringJoinsave_12/Constsave_12/StringJoin/inputs_1*
	separator *
N*
_output_shapes
: 
T
save_12/num_shardsConst*
_output_shapes
: *
value	B :*
dtype0
_
save_12/ShardedFilename/shardConst*
_output_shapes
: *
dtype0*
value	B : 
Й
save_12/ShardedFilenameShardedFilenamesave_12/StringJoinsave_12/ShardedFilename/shardsave_12/num_shards*
_output_shapes
: 
Л
save_12/SaveV2/tensor_namesConst*
_output_shapes
:(*ї
value±BЃ(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
dtype0
ґ
save_12/SaveV2/shape_and_slicesConst*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:(
Ќ
save_12/SaveV2SaveV2save_12/ShardedFilenamesave_12/SaveV2/tensor_namessave_12/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1v/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1v/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1v/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1v/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1v/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1v/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1*6
dtypes,
*2(
Э
save_12/control_dependencyIdentitysave_12/ShardedFilename^save_12/SaveV2*
_output_shapes
: **
_class 
loc:@save_12/ShardedFilename*
T0
¶
.save_12/MergeV2Checkpoints/checkpoint_prefixesPacksave_12/ShardedFilename^save_12/control_dependency*

axis *
_output_shapes
:*
T0*
N
Ж
save_12/MergeV2CheckpointsMergeV2Checkpoints.save_12/MergeV2Checkpoints/checkpoint_prefixessave_12/Const*
delete_old_dirs(
Ж
save_12/IdentityIdentitysave_12/Const^save_12/MergeV2Checkpoints^save_12/control_dependency*
_output_shapes
: *
T0
О
save_12/RestoreV2/tensor_namesConst*
_output_shapes
:(*
dtype0*ї
value±BЃ(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1
є
"save_12/RestoreV2/shape_and_slicesConst*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:(
в
save_12/RestoreV2	RestoreV2save_12/Constsave_12/RestoreV2/tensor_names"save_12/RestoreV2/shape_and_slices*6
dtypes,
*2(*ґ
_output_shapes£
†::::::::::::::::::::::::::::::::::::::::
§
save_12/AssignAssignbeta1_powersave_12/RestoreV2*
validate_shape(*
use_locking(* 
_class
loc:@pi/dense/bias*
T0*
_output_shapes
: 
©
save_12/Assign_1Assignbeta1_power_1save_12/RestoreV2:1*
validate_shape(*
use_locking(*
T0*
_output_shapes
: *
_class
loc:@v/dense/bias
®
save_12/Assign_2Assignbeta2_powersave_12/RestoreV2:2*
T0* 
_class
loc:@pi/dense/bias*
use_locking(*
_output_shapes
: *
validate_shape(
©
save_12/Assign_3Assignbeta2_power_1save_12/RestoreV2:3*
_output_shapes
: *
_class
loc:@v/dense/bias*
T0*
use_locking(*
validate_shape(
Ѓ
save_12/Assign_4Assignpi/dense/biassave_12/RestoreV2:4*
validate_shape(*
use_locking(* 
_class
loc:@pi/dense/bias*
T0*
_output_shapes
:@
≥
save_12/Assign_5Assignpi/dense/bias/Adamsave_12/RestoreV2:5*
use_locking(* 
_class
loc:@pi/dense/bias*
validate_shape(*
T0*
_output_shapes
:@
µ
save_12/Assign_6Assignpi/dense/bias/Adam_1save_12/RestoreV2:6*
use_locking(*
_output_shapes
:@*
validate_shape(* 
_class
loc:@pi/dense/bias*
T0
ґ
save_12/Assign_7Assignpi/dense/kernelsave_12/RestoreV2:7*
_output_shapes

:@*"
_class
loc:@pi/dense/kernel*
validate_shape(*
T0*
use_locking(
ї
save_12/Assign_8Assignpi/dense/kernel/Adamsave_12/RestoreV2:8*
T0*
validate_shape(*
use_locking(*"
_class
loc:@pi/dense/kernel*
_output_shapes

:@
љ
save_12/Assign_9Assignpi/dense/kernel/Adam_1save_12/RestoreV2:9*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes

:@
і
save_12/Assign_10Assignpi/dense_1/biassave_12/RestoreV2:10*"
_class
loc:@pi/dense_1/bias*
T0*
validate_shape(*
use_locking(*
_output_shapes
:@
є
save_12/Assign_11Assignpi/dense_1/bias/Adamsave_12/RestoreV2:11*"
_class
loc:@pi/dense_1/bias*
use_locking(*
T0*
validate_shape(*
_output_shapes
:@
ї
save_12/Assign_12Assignpi/dense_1/bias/Adam_1save_12/RestoreV2:12*
_output_shapes
:@*
use_locking(*
validate_shape(*
T0*"
_class
loc:@pi/dense_1/bias
Љ
save_12/Assign_13Assignpi/dense_1/kernelsave_12/RestoreV2:13*
T0*
_output_shapes

:@@*
validate_shape(*$
_class
loc:@pi/dense_1/kernel*
use_locking(
Ѕ
save_12/Assign_14Assignpi/dense_1/kernel/Adamsave_12/RestoreV2:14*$
_class
loc:@pi/dense_1/kernel*
use_locking(*
validate_shape(*
_output_shapes

:@@*
T0
√
save_12/Assign_15Assignpi/dense_1/kernel/Adam_1save_12/RestoreV2:15*$
_class
loc:@pi/dense_1/kernel*
use_locking(*
validate_shape(*
T0*
_output_shapes

:@@
і
save_12/Assign_16Assignpi/dense_2/biassave_12/RestoreV2:16*"
_class
loc:@pi/dense_2/bias*
use_locking(*
validate_shape(*
T0*
_output_shapes
:
є
save_12/Assign_17Assignpi/dense_2/bias/Adamsave_12/RestoreV2:17*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
T0*
use_locking(*
_output_shapes
:
ї
save_12/Assign_18Assignpi/dense_2/bias/Adam_1save_12/RestoreV2:18*
use_locking(*
_output_shapes
:*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
T0
Љ
save_12/Assign_19Assignpi/dense_2/kernelsave_12/RestoreV2:19*$
_class
loc:@pi/dense_2/kernel*
T0*
use_locking(*
validate_shape(*
_output_shapes

:@
Ѕ
save_12/Assign_20Assignpi/dense_2/kernel/Adamsave_12/RestoreV2:20*$
_class
loc:@pi/dense_2/kernel*
_output_shapes

:@*
validate_shape(*
T0*
use_locking(
√
save_12/Assign_21Assignpi/dense_2/kernel/Adam_1save_12/RestoreV2:21*
use_locking(*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@*
T0
Ѓ
save_12/Assign_22Assignv/dense/biassave_12/RestoreV2:22*
T0*
use_locking(*
_output_shapes
:@*
_class
loc:@v/dense/bias*
validate_shape(
≥
save_12/Assign_23Assignv/dense/bias/Adamsave_12/RestoreV2:23*
_output_shapes
:@*
T0*
use_locking(*
validate_shape(*
_class
loc:@v/dense/bias
µ
save_12/Assign_24Assignv/dense/bias/Adam_1save_12/RestoreV2:24*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0
ґ
save_12/Assign_25Assignv/dense/kernelsave_12/RestoreV2:25*
validate_shape(*
T0*
_output_shapes

:@*
use_locking(*!
_class
loc:@v/dense/kernel
ї
save_12/Assign_26Assignv/dense/kernel/Adamsave_12/RestoreV2:26*!
_class
loc:@v/dense/kernel*
use_locking(*
_output_shapes

:@*
T0*
validate_shape(
љ
save_12/Assign_27Assignv/dense/kernel/Adam_1save_12/RestoreV2:27*
T0*
validate_shape(*!
_class
loc:@v/dense/kernel*
use_locking(*
_output_shapes

:@
≤
save_12/Assign_28Assignv/dense_1/biassave_12/RestoreV2:28*
use_locking(*
T0*
_output_shapes
:@*!
_class
loc:@v/dense_1/bias*
validate_shape(
Ј
save_12/Assign_29Assignv/dense_1/bias/Adamsave_12/RestoreV2:29*
_output_shapes
:@*
T0*!
_class
loc:@v/dense_1/bias*
validate_shape(*
use_locking(
є
save_12/Assign_30Assignv/dense_1/bias/Adam_1save_12/RestoreV2:30*!
_class
loc:@v/dense_1/bias*
T0*
_output_shapes
:@*
validate_shape(*
use_locking(
Ї
save_12/Assign_31Assignv/dense_1/kernelsave_12/RestoreV2:31*
_output_shapes

:@@*
use_locking(*
T0*
validate_shape(*#
_class
loc:@v/dense_1/kernel
њ
save_12/Assign_32Assignv/dense_1/kernel/Adamsave_12/RestoreV2:32*
T0*
_output_shapes

:@@*
use_locking(*#
_class
loc:@v/dense_1/kernel*
validate_shape(
Ѕ
save_12/Assign_33Assignv/dense_1/kernel/Adam_1save_12/RestoreV2:33*#
_class
loc:@v/dense_1/kernel*
validate_shape(*
use_locking(*
T0*
_output_shapes

:@@
≤
save_12/Assign_34Assignv/dense_2/biassave_12/RestoreV2:34*
use_locking(*
validate_shape(*!
_class
loc:@v/dense_2/bias*
_output_shapes
:*
T0
Ј
save_12/Assign_35Assignv/dense_2/bias/Adamsave_12/RestoreV2:35*
use_locking(*
T0*
validate_shape(*!
_class
loc:@v/dense_2/bias*
_output_shapes
:
є
save_12/Assign_36Assignv/dense_2/bias/Adam_1save_12/RestoreV2:36*
use_locking(*
validate_shape(*!
_class
loc:@v/dense_2/bias*
T0*
_output_shapes
:
Ї
save_12/Assign_37Assignv/dense_2/kernelsave_12/RestoreV2:37*
_output_shapes

:@*
validate_shape(*#
_class
loc:@v/dense_2/kernel*
T0*
use_locking(
њ
save_12/Assign_38Assignv/dense_2/kernel/Adamsave_12/RestoreV2:38*
validate_shape(*
T0*
_output_shapes

:@*#
_class
loc:@v/dense_2/kernel*
use_locking(
Ѕ
save_12/Assign_39Assignv/dense_2/kernel/Adam_1save_12/RestoreV2:39*
_output_shapes

:@*
use_locking(*#
_class
loc:@v/dense_2/kernel*
T0*
validate_shape(
±
save_12/restore_shardNoOp^save_12/Assign^save_12/Assign_1^save_12/Assign_10^save_12/Assign_11^save_12/Assign_12^save_12/Assign_13^save_12/Assign_14^save_12/Assign_15^save_12/Assign_16^save_12/Assign_17^save_12/Assign_18^save_12/Assign_19^save_12/Assign_2^save_12/Assign_20^save_12/Assign_21^save_12/Assign_22^save_12/Assign_23^save_12/Assign_24^save_12/Assign_25^save_12/Assign_26^save_12/Assign_27^save_12/Assign_28^save_12/Assign_29^save_12/Assign_3^save_12/Assign_30^save_12/Assign_31^save_12/Assign_32^save_12/Assign_33^save_12/Assign_34^save_12/Assign_35^save_12/Assign_36^save_12/Assign_37^save_12/Assign_38^save_12/Assign_39^save_12/Assign_4^save_12/Assign_5^save_12/Assign_6^save_12/Assign_7^save_12/Assign_8^save_12/Assign_9
3
save_12/restore_allNoOp^save_12/restore_shard
\
save_13/filename/inputConst*
valueB Bmodel*
_output_shapes
: *
dtype0
t
save_13/filenamePlaceholderWithDefaultsave_13/filename/input*
dtype0*
shape: *
_output_shapes
: 
k
save_13/ConstPlaceholderWithDefaultsave_13/filename*
_output_shapes
: *
dtype0*
shape: 
З
save_13/StringJoin/inputs_1Const*<
value3B1 B+_temp_e42cfbf4cbc841438465a20759e55934/part*
dtype0*
_output_shapes
: 
~
save_13/StringJoin
StringJoinsave_13/Constsave_13/StringJoin/inputs_1*
	separator *
_output_shapes
: *
N
T
save_13/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
_
save_13/ShardedFilename/shardConst*
dtype0*
_output_shapes
: *
value	B : 
Й
save_13/ShardedFilenameShardedFilenamesave_13/StringJoinsave_13/ShardedFilename/shardsave_13/num_shards*
_output_shapes
: 
Л
save_13/SaveV2/tensor_namesConst*
dtype0*ї
value±BЃ(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
_output_shapes
:(
ґ
save_13/SaveV2/shape_and_slicesConst*
_output_shapes
:(*
dtype0*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
Ќ
save_13/SaveV2SaveV2save_13/ShardedFilenamesave_13/SaveV2/tensor_namessave_13/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1v/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1v/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1v/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1v/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1v/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1v/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1*6
dtypes,
*2(
Э
save_13/control_dependencyIdentitysave_13/ShardedFilename^save_13/SaveV2*
_output_shapes
: **
_class 
loc:@save_13/ShardedFilename*
T0
¶
.save_13/MergeV2Checkpoints/checkpoint_prefixesPacksave_13/ShardedFilename^save_13/control_dependency*
N*
T0*

axis *
_output_shapes
:
Ж
save_13/MergeV2CheckpointsMergeV2Checkpoints.save_13/MergeV2Checkpoints/checkpoint_prefixessave_13/Const*
delete_old_dirs(
Ж
save_13/IdentityIdentitysave_13/Const^save_13/MergeV2Checkpoints^save_13/control_dependency*
_output_shapes
: *
T0
О
save_13/RestoreV2/tensor_namesConst*ї
value±BЃ(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:(
є
"save_13/RestoreV2/shape_and_slicesConst*
_output_shapes
:(*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
в
save_13/RestoreV2	RestoreV2save_13/Constsave_13/RestoreV2/tensor_names"save_13/RestoreV2/shape_and_slices*6
dtypes,
*2(*ґ
_output_shapes£
†::::::::::::::::::::::::::::::::::::::::
§
save_13/AssignAssignbeta1_powersave_13/RestoreV2*
use_locking(*
validate_shape(*
T0*
_output_shapes
: * 
_class
loc:@pi/dense/bias
©
save_13/Assign_1Assignbeta1_power_1save_13/RestoreV2:1*
_output_shapes
: *
validate_shape(*
T0*
_class
loc:@v/dense/bias*
use_locking(
®
save_13/Assign_2Assignbeta2_powersave_13/RestoreV2:2*
T0*
_output_shapes
: *
use_locking(* 
_class
loc:@pi/dense/bias*
validate_shape(
©
save_13/Assign_3Assignbeta2_power_1save_13/RestoreV2:3*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
Ѓ
save_13/Assign_4Assignpi/dense/biassave_13/RestoreV2:4*
_output_shapes
:@* 
_class
loc:@pi/dense/bias*
T0*
use_locking(*
validate_shape(
≥
save_13/Assign_5Assignpi/dense/bias/Adamsave_13/RestoreV2:5*
use_locking(*
validate_shape(* 
_class
loc:@pi/dense/bias*
T0*
_output_shapes
:@
µ
save_13/Assign_6Assignpi/dense/bias/Adam_1save_13/RestoreV2:6*
use_locking(*
_output_shapes
:@*
T0*
validate_shape(* 
_class
loc:@pi/dense/bias
ґ
save_13/Assign_7Assignpi/dense/kernelsave_13/RestoreV2:7*
T0*
use_locking(*"
_class
loc:@pi/dense/kernel*
_output_shapes

:@*
validate_shape(
ї
save_13/Assign_8Assignpi/dense/kernel/Adamsave_13/RestoreV2:8*
use_locking(*
validate_shape(*
_output_shapes

:@*
T0*"
_class
loc:@pi/dense/kernel
љ
save_13/Assign_9Assignpi/dense/kernel/Adam_1save_13/RestoreV2:9*
validate_shape(*
T0*
_output_shapes

:@*"
_class
loc:@pi/dense/kernel*
use_locking(
і
save_13/Assign_10Assignpi/dense_1/biassave_13/RestoreV2:10*
validate_shape(*
use_locking(*
T0*"
_class
loc:@pi/dense_1/bias*
_output_shapes
:@
є
save_13/Assign_11Assignpi/dense_1/bias/Adamsave_13/RestoreV2:11*
use_locking(*
T0*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
_output_shapes
:@
ї
save_13/Assign_12Assignpi/dense_1/bias/Adam_1save_13/RestoreV2:12*
T0*
use_locking(*
validate_shape(*"
_class
loc:@pi/dense_1/bias*
_output_shapes
:@
Љ
save_13/Assign_13Assignpi/dense_1/kernelsave_13/RestoreV2:13*$
_class
loc:@pi/dense_1/kernel*
use_locking(*
_output_shapes

:@@*
validate_shape(*
T0
Ѕ
save_13/Assign_14Assignpi/dense_1/kernel/Adamsave_13/RestoreV2:14*
_output_shapes

:@@*
T0*$
_class
loc:@pi/dense_1/kernel*
use_locking(*
validate_shape(
√
save_13/Assign_15Assignpi/dense_1/kernel/Adam_1save_13/RestoreV2:15*$
_class
loc:@pi/dense_1/kernel*
T0*
use_locking(*
_output_shapes

:@@*
validate_shape(
і
save_13/Assign_16Assignpi/dense_2/biassave_13/RestoreV2:16*
use_locking(*
_output_shapes
:*"
_class
loc:@pi/dense_2/bias*
T0*
validate_shape(
є
save_13/Assign_17Assignpi/dense_2/bias/Adamsave_13/RestoreV2:17*
validate_shape(*
_output_shapes
:*
T0*
use_locking(*"
_class
loc:@pi/dense_2/bias
ї
save_13/Assign_18Assignpi/dense_2/bias/Adam_1save_13/RestoreV2:18*
T0*
use_locking(*
_output_shapes
:*"
_class
loc:@pi/dense_2/bias*
validate_shape(
Љ
save_13/Assign_19Assignpi/dense_2/kernelsave_13/RestoreV2:19*
_output_shapes

:@*
T0*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
use_locking(
Ѕ
save_13/Assign_20Assignpi/dense_2/kernel/Adamsave_13/RestoreV2:20*
use_locking(*
validate_shape(*
T0*
_output_shapes

:@*$
_class
loc:@pi/dense_2/kernel
√
save_13/Assign_21Assignpi/dense_2/kernel/Adam_1save_13/RestoreV2:21*
validate_shape(*$
_class
loc:@pi/dense_2/kernel*
T0*
_output_shapes

:@*
use_locking(
Ѓ
save_13/Assign_22Assignv/dense/biassave_13/RestoreV2:22*
validate_shape(*
_output_shapes
:@*
use_locking(*
_class
loc:@v/dense/bias*
T0
≥
save_13/Assign_23Assignv/dense/bias/Adamsave_13/RestoreV2:23*
T0*
validate_shape(*
use_locking(*
_output_shapes
:@*
_class
loc:@v/dense/bias
µ
save_13/Assign_24Assignv/dense/bias/Adam_1save_13/RestoreV2:24*
use_locking(*
validate_shape(*
_class
loc:@v/dense/bias*
T0*
_output_shapes
:@
ґ
save_13/Assign_25Assignv/dense/kernelsave_13/RestoreV2:25*!
_class
loc:@v/dense/kernel*
validate_shape(*
use_locking(*
T0*
_output_shapes

:@
ї
save_13/Assign_26Assignv/dense/kernel/Adamsave_13/RestoreV2:26*
_output_shapes

:@*!
_class
loc:@v/dense/kernel*
T0*
use_locking(*
validate_shape(
љ
save_13/Assign_27Assignv/dense/kernel/Adam_1save_13/RestoreV2:27*
_output_shapes

:@*
T0*
validate_shape(*!
_class
loc:@v/dense/kernel*
use_locking(
≤
save_13/Assign_28Assignv/dense_1/biassave_13/RestoreV2:28*
_output_shapes
:@*
validate_shape(*
use_locking(*!
_class
loc:@v/dense_1/bias*
T0
Ј
save_13/Assign_29Assignv/dense_1/bias/Adamsave_13/RestoreV2:29*!
_class
loc:@v/dense_1/bias*
validate_shape(*
T0*
_output_shapes
:@*
use_locking(
є
save_13/Assign_30Assignv/dense_1/bias/Adam_1save_13/RestoreV2:30*
_output_shapes
:@*!
_class
loc:@v/dense_1/bias*
validate_shape(*
use_locking(*
T0
Ї
save_13/Assign_31Assignv/dense_1/kernelsave_13/RestoreV2:31*
T0*
_output_shapes

:@@*#
_class
loc:@v/dense_1/kernel*
validate_shape(*
use_locking(
њ
save_13/Assign_32Assignv/dense_1/kernel/Adamsave_13/RestoreV2:32*
use_locking(*
validate_shape(*#
_class
loc:@v/dense_1/kernel*
_output_shapes

:@@*
T0
Ѕ
save_13/Assign_33Assignv/dense_1/kernel/Adam_1save_13/RestoreV2:33*
T0*
_output_shapes

:@@*#
_class
loc:@v/dense_1/kernel*
validate_shape(*
use_locking(
≤
save_13/Assign_34Assignv/dense_2/biassave_13/RestoreV2:34*
use_locking(*!
_class
loc:@v/dense_2/bias*
validate_shape(*
T0*
_output_shapes
:
Ј
save_13/Assign_35Assignv/dense_2/bias/Adamsave_13/RestoreV2:35*!
_class
loc:@v/dense_2/bias*
_output_shapes
:*
use_locking(*
T0*
validate_shape(
є
save_13/Assign_36Assignv/dense_2/bias/Adam_1save_13/RestoreV2:36*
_output_shapes
:*!
_class
loc:@v/dense_2/bias*
use_locking(*
validate_shape(*
T0
Ї
save_13/Assign_37Assignv/dense_2/kernelsave_13/RestoreV2:37*#
_class
loc:@v/dense_2/kernel*
use_locking(*
_output_shapes

:@*
T0*
validate_shape(
њ
save_13/Assign_38Assignv/dense_2/kernel/Adamsave_13/RestoreV2:38*
validate_shape(*
use_locking(*
T0*#
_class
loc:@v/dense_2/kernel*
_output_shapes

:@
Ѕ
save_13/Assign_39Assignv/dense_2/kernel/Adam_1save_13/RestoreV2:39*
use_locking(*#
_class
loc:@v/dense_2/kernel*
T0*
_output_shapes

:@*
validate_shape(
±
save_13/restore_shardNoOp^save_13/Assign^save_13/Assign_1^save_13/Assign_10^save_13/Assign_11^save_13/Assign_12^save_13/Assign_13^save_13/Assign_14^save_13/Assign_15^save_13/Assign_16^save_13/Assign_17^save_13/Assign_18^save_13/Assign_19^save_13/Assign_2^save_13/Assign_20^save_13/Assign_21^save_13/Assign_22^save_13/Assign_23^save_13/Assign_24^save_13/Assign_25^save_13/Assign_26^save_13/Assign_27^save_13/Assign_28^save_13/Assign_29^save_13/Assign_3^save_13/Assign_30^save_13/Assign_31^save_13/Assign_32^save_13/Assign_33^save_13/Assign_34^save_13/Assign_35^save_13/Assign_36^save_13/Assign_37^save_13/Assign_38^save_13/Assign_39^save_13/Assign_4^save_13/Assign_5^save_13/Assign_6^save_13/Assign_7^save_13/Assign_8^save_13/Assign_9
3
save_13/restore_allNoOp^save_13/restore_shard
\
save_14/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
t
save_14/filenamePlaceholderWithDefaultsave_14/filename/input*
shape: *
dtype0*
_output_shapes
: 
k
save_14/ConstPlaceholderWithDefaultsave_14/filename*
_output_shapes
: *
shape: *
dtype0
З
save_14/StringJoin/inputs_1Const*<
value3B1 B+_temp_2dfba88ec1c940faaeb002e1d8bcabbb/part*
_output_shapes
: *
dtype0
~
save_14/StringJoin
StringJoinsave_14/Constsave_14/StringJoin/inputs_1*
	separator *
N*
_output_shapes
: 
T
save_14/num_shardsConst*
dtype0*
_output_shapes
: *
value	B :
_
save_14/ShardedFilename/shardConst*
value	B : *
_output_shapes
: *
dtype0
Й
save_14/ShardedFilenameShardedFilenamesave_14/StringJoinsave_14/ShardedFilename/shardsave_14/num_shards*
_output_shapes
: 
Л
save_14/SaveV2/tensor_namesConst*ї
value±BЃ(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
_output_shapes
:(*
dtype0
ґ
save_14/SaveV2/shape_and_slicesConst*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:(
Ќ
save_14/SaveV2SaveV2save_14/ShardedFilenamesave_14/SaveV2/tensor_namessave_14/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1v/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1v/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1v/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1v/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1v/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1v/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1*6
dtypes,
*2(
Э
save_14/control_dependencyIdentitysave_14/ShardedFilename^save_14/SaveV2*
T0**
_class 
loc:@save_14/ShardedFilename*
_output_shapes
: 
¶
.save_14/MergeV2Checkpoints/checkpoint_prefixesPacksave_14/ShardedFilename^save_14/control_dependency*
N*
_output_shapes
:*
T0*

axis 
Ж
save_14/MergeV2CheckpointsMergeV2Checkpoints.save_14/MergeV2Checkpoints/checkpoint_prefixessave_14/Const*
delete_old_dirs(
Ж
save_14/IdentityIdentitysave_14/Const^save_14/MergeV2Checkpoints^save_14/control_dependency*
_output_shapes
: *
T0
О
save_14/RestoreV2/tensor_namesConst*ї
value±BЃ(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:(
є
"save_14/RestoreV2/shape_and_slicesConst*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:(
в
save_14/RestoreV2	RestoreV2save_14/Constsave_14/RestoreV2/tensor_names"save_14/RestoreV2/shape_and_slices*ґ
_output_shapes£
†::::::::::::::::::::::::::::::::::::::::*6
dtypes,
*2(
§
save_14/AssignAssignbeta1_powersave_14/RestoreV2*
validate_shape(*
use_locking(*
T0*
_output_shapes
: * 
_class
loc:@pi/dense/bias
©
save_14/Assign_1Assignbeta1_power_1save_14/RestoreV2:1*
use_locking(*
_class
loc:@v/dense/bias*
_output_shapes
: *
validate_shape(*
T0
®
save_14/Assign_2Assignbeta2_powersave_14/RestoreV2:2*
validate_shape(* 
_class
loc:@pi/dense/bias*
use_locking(*
_output_shapes
: *
T0
©
save_14/Assign_3Assignbeta2_power_1save_14/RestoreV2:3*
_class
loc:@v/dense/bias*
use_locking(*
validate_shape(*
T0*
_output_shapes
: 
Ѓ
save_14/Assign_4Assignpi/dense/biassave_14/RestoreV2:4* 
_class
loc:@pi/dense/bias*
use_locking(*
T0*
validate_shape(*
_output_shapes
:@
≥
save_14/Assign_5Assignpi/dense/bias/Adamsave_14/RestoreV2:5*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
_output_shapes
:@*
validate_shape(
µ
save_14/Assign_6Assignpi/dense/bias/Adam_1save_14/RestoreV2:6*
T0* 
_class
loc:@pi/dense/bias*
use_locking(*
validate_shape(*
_output_shapes
:@
ґ
save_14/Assign_7Assignpi/dense/kernelsave_14/RestoreV2:7*
_output_shapes

:@*"
_class
loc:@pi/dense/kernel*
use_locking(*
validate_shape(*
T0
ї
save_14/Assign_8Assignpi/dense/kernel/Adamsave_14/RestoreV2:8*
validate_shape(*"
_class
loc:@pi/dense/kernel*
T0*
use_locking(*
_output_shapes

:@
љ
save_14/Assign_9Assignpi/dense/kernel/Adam_1save_14/RestoreV2:9*
T0*
validate_shape(*
use_locking(*"
_class
loc:@pi/dense/kernel*
_output_shapes

:@
і
save_14/Assign_10Assignpi/dense_1/biassave_14/RestoreV2:10*
_output_shapes
:@*"
_class
loc:@pi/dense_1/bias*
T0*
validate_shape(*
use_locking(
є
save_14/Assign_11Assignpi/dense_1/bias/Adamsave_14/RestoreV2:11*"
_class
loc:@pi/dense_1/bias*
_output_shapes
:@*
T0*
validate_shape(*
use_locking(
ї
save_14/Assign_12Assignpi/dense_1/bias/Adam_1save_14/RestoreV2:12*"
_class
loc:@pi/dense_1/bias*
_output_shapes
:@*
T0*
use_locking(*
validate_shape(
Љ
save_14/Assign_13Assignpi/dense_1/kernelsave_14/RestoreV2:13*
_output_shapes

:@@*$
_class
loc:@pi/dense_1/kernel*
T0*
use_locking(*
validate_shape(
Ѕ
save_14/Assign_14Assignpi/dense_1/kernel/Adamsave_14/RestoreV2:14*
T0*
_output_shapes

:@@*
use_locking(*
validate_shape(*$
_class
loc:@pi/dense_1/kernel
√
save_14/Assign_15Assignpi/dense_1/kernel/Adam_1save_14/RestoreV2:15*$
_class
loc:@pi/dense_1/kernel*
use_locking(*
validate_shape(*
T0*
_output_shapes

:@@
і
save_14/Assign_16Assignpi/dense_2/biassave_14/RestoreV2:16*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
T0*
_output_shapes
:*
use_locking(
є
save_14/Assign_17Assignpi/dense_2/bias/Adamsave_14/RestoreV2:17*
_output_shapes
:*
T0*
validate_shape(*
use_locking(*"
_class
loc:@pi/dense_2/bias
ї
save_14/Assign_18Assignpi/dense_2/bias/Adam_1save_14/RestoreV2:18*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:*
use_locking(*
validate_shape(*
T0
Љ
save_14/Assign_19Assignpi/dense_2/kernelsave_14/RestoreV2:19*
use_locking(*
validate_shape(*
T0*
_output_shapes

:@*$
_class
loc:@pi/dense_2/kernel
Ѕ
save_14/Assign_20Assignpi/dense_2/kernel/Adamsave_14/RestoreV2:20*
validate_shape(*$
_class
loc:@pi/dense_2/kernel*
use_locking(*
T0*
_output_shapes

:@
√
save_14/Assign_21Assignpi/dense_2/kernel/Adam_1save_14/RestoreV2:21*
use_locking(*
validate_shape(*
_output_shapes

:@*$
_class
loc:@pi/dense_2/kernel*
T0
Ѓ
save_14/Assign_22Assignv/dense/biassave_14/RestoreV2:22*
_output_shapes
:@*
validate_shape(*
use_locking(*
_class
loc:@v/dense/bias*
T0
≥
save_14/Assign_23Assignv/dense/bias/Adamsave_14/RestoreV2:23*
validate_shape(*
T0*
_output_shapes
:@*
_class
loc:@v/dense/bias*
use_locking(
µ
save_14/Assign_24Assignv/dense/bias/Adam_1save_14/RestoreV2:24*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*
_class
loc:@v/dense/bias
ґ
save_14/Assign_25Assignv/dense/kernelsave_14/RestoreV2:25*!
_class
loc:@v/dense/kernel*
_output_shapes

:@*
use_locking(*
T0*
validate_shape(
ї
save_14/Assign_26Assignv/dense/kernel/Adamsave_14/RestoreV2:26*
_output_shapes

:@*
validate_shape(*
T0*!
_class
loc:@v/dense/kernel*
use_locking(
љ
save_14/Assign_27Assignv/dense/kernel/Adam_1save_14/RestoreV2:27*!
_class
loc:@v/dense/kernel*
T0*
_output_shapes

:@*
use_locking(*
validate_shape(
≤
save_14/Assign_28Assignv/dense_1/biassave_14/RestoreV2:28*!
_class
loc:@v/dense_1/bias*
T0*
use_locking(*
_output_shapes
:@*
validate_shape(
Ј
save_14/Assign_29Assignv/dense_1/bias/Adamsave_14/RestoreV2:29*!
_class
loc:@v/dense_1/bias*
T0*
validate_shape(*
_output_shapes
:@*
use_locking(
є
save_14/Assign_30Assignv/dense_1/bias/Adam_1save_14/RestoreV2:30*
T0*
_output_shapes
:@*!
_class
loc:@v/dense_1/bias*
use_locking(*
validate_shape(
Ї
save_14/Assign_31Assignv/dense_1/kernelsave_14/RestoreV2:31*
validate_shape(*
_output_shapes

:@@*
T0*#
_class
loc:@v/dense_1/kernel*
use_locking(
њ
save_14/Assign_32Assignv/dense_1/kernel/Adamsave_14/RestoreV2:32*#
_class
loc:@v/dense_1/kernel*
_output_shapes

:@@*
T0*
use_locking(*
validate_shape(
Ѕ
save_14/Assign_33Assignv/dense_1/kernel/Adam_1save_14/RestoreV2:33*
_output_shapes

:@@*#
_class
loc:@v/dense_1/kernel*
use_locking(*
T0*
validate_shape(
≤
save_14/Assign_34Assignv/dense_2/biassave_14/RestoreV2:34*
validate_shape(*!
_class
loc:@v/dense_2/bias*
_output_shapes
:*
use_locking(*
T0
Ј
save_14/Assign_35Assignv/dense_2/bias/Adamsave_14/RestoreV2:35*!
_class
loc:@v/dense_2/bias*
use_locking(*
_output_shapes
:*
T0*
validate_shape(
є
save_14/Assign_36Assignv/dense_2/bias/Adam_1save_14/RestoreV2:36*
T0*
validate_shape(*
use_locking(*
_output_shapes
:*!
_class
loc:@v/dense_2/bias
Ї
save_14/Assign_37Assignv/dense_2/kernelsave_14/RestoreV2:37*#
_class
loc:@v/dense_2/kernel*
use_locking(*
_output_shapes

:@*
T0*
validate_shape(
њ
save_14/Assign_38Assignv/dense_2/kernel/Adamsave_14/RestoreV2:38*
validate_shape(*
use_locking(*
T0*
_output_shapes

:@*#
_class
loc:@v/dense_2/kernel
Ѕ
save_14/Assign_39Assignv/dense_2/kernel/Adam_1save_14/RestoreV2:39*
use_locking(*#
_class
loc:@v/dense_2/kernel*
T0*
_output_shapes

:@*
validate_shape(
±
save_14/restore_shardNoOp^save_14/Assign^save_14/Assign_1^save_14/Assign_10^save_14/Assign_11^save_14/Assign_12^save_14/Assign_13^save_14/Assign_14^save_14/Assign_15^save_14/Assign_16^save_14/Assign_17^save_14/Assign_18^save_14/Assign_19^save_14/Assign_2^save_14/Assign_20^save_14/Assign_21^save_14/Assign_22^save_14/Assign_23^save_14/Assign_24^save_14/Assign_25^save_14/Assign_26^save_14/Assign_27^save_14/Assign_28^save_14/Assign_29^save_14/Assign_3^save_14/Assign_30^save_14/Assign_31^save_14/Assign_32^save_14/Assign_33^save_14/Assign_34^save_14/Assign_35^save_14/Assign_36^save_14/Assign_37^save_14/Assign_38^save_14/Assign_39^save_14/Assign_4^save_14/Assign_5^save_14/Assign_6^save_14/Assign_7^save_14/Assign_8^save_14/Assign_9
3
save_14/restore_allNoOp^save_14/restore_shard
\
save_15/filename/inputConst*
_output_shapes
: *
dtype0*
valueB Bmodel
t
save_15/filenamePlaceholderWithDefaultsave_15/filename/input*
_output_shapes
: *
shape: *
dtype0
k
save_15/ConstPlaceholderWithDefaultsave_15/filename*
_output_shapes
: *
dtype0*
shape: 
З
save_15/StringJoin/inputs_1Const*<
value3B1 B+_temp_88bb901a8ed443a087e744afd862ab16/part*
dtype0*
_output_shapes
: 
~
save_15/StringJoin
StringJoinsave_15/Constsave_15/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
T
save_15/num_shardsConst*
value	B :*
_output_shapes
: *
dtype0
_
save_15/ShardedFilename/shardConst*
dtype0*
value	B : *
_output_shapes
: 
Й
save_15/ShardedFilenameShardedFilenamesave_15/StringJoinsave_15/ShardedFilename/shardsave_15/num_shards*
_output_shapes
: 
Л
save_15/SaveV2/tensor_namesConst*
_output_shapes
:(*
dtype0*ї
value±BЃ(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1
ґ
save_15/SaveV2/shape_and_slicesConst*
dtype0*
_output_shapes
:(*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
Ќ
save_15/SaveV2SaveV2save_15/ShardedFilenamesave_15/SaveV2/tensor_namessave_15/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1v/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1v/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1v/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1v/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1v/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1v/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1*6
dtypes,
*2(
Э
save_15/control_dependencyIdentitysave_15/ShardedFilename^save_15/SaveV2*
_output_shapes
: **
_class 
loc:@save_15/ShardedFilename*
T0
¶
.save_15/MergeV2Checkpoints/checkpoint_prefixesPacksave_15/ShardedFilename^save_15/control_dependency*
N*

axis *
_output_shapes
:*
T0
Ж
save_15/MergeV2CheckpointsMergeV2Checkpoints.save_15/MergeV2Checkpoints/checkpoint_prefixessave_15/Const*
delete_old_dirs(
Ж
save_15/IdentityIdentitysave_15/Const^save_15/MergeV2Checkpoints^save_15/control_dependency*
T0*
_output_shapes
: 
О
save_15/RestoreV2/tensor_namesConst*
dtype0*
_output_shapes
:(*ї
value±BЃ(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1
є
"save_15/RestoreV2/shape_and_slicesConst*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:(*
dtype0
в
save_15/RestoreV2	RestoreV2save_15/Constsave_15/RestoreV2/tensor_names"save_15/RestoreV2/shape_and_slices*6
dtypes,
*2(*ґ
_output_shapes£
†::::::::::::::::::::::::::::::::::::::::
§
save_15/AssignAssignbeta1_powersave_15/RestoreV2*
validate_shape(* 
_class
loc:@pi/dense/bias*
_output_shapes
: *
T0*
use_locking(
©
save_15/Assign_1Assignbeta1_power_1save_15/RestoreV2:1*
use_locking(*
T0*
_output_shapes
: *
_class
loc:@v/dense/bias*
validate_shape(
®
save_15/Assign_2Assignbeta2_powersave_15/RestoreV2:2*
T0*
_output_shapes
: *
use_locking(* 
_class
loc:@pi/dense/bias*
validate_shape(
©
save_15/Assign_3Assignbeta2_power_1save_15/RestoreV2:3*
validate_shape(*
use_locking(*
T0*
_class
loc:@v/dense/bias*
_output_shapes
: 
Ѓ
save_15/Assign_4Assignpi/dense/biassave_15/RestoreV2:4* 
_class
loc:@pi/dense/bias*
_output_shapes
:@*
T0*
use_locking(*
validate_shape(
≥
save_15/Assign_5Assignpi/dense/bias/Adamsave_15/RestoreV2:5*
use_locking(* 
_class
loc:@pi/dense/bias*
_output_shapes
:@*
T0*
validate_shape(
µ
save_15/Assign_6Assignpi/dense/bias/Adam_1save_15/RestoreV2:6*
T0*
_output_shapes
:@*
validate_shape(* 
_class
loc:@pi/dense/bias*
use_locking(
ґ
save_15/Assign_7Assignpi/dense/kernelsave_15/RestoreV2:7*
use_locking(*
_output_shapes

:@*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(
ї
save_15/Assign_8Assignpi/dense/kernel/Adamsave_15/RestoreV2:8*"
_class
loc:@pi/dense/kernel*
T0*
use_locking(*
_output_shapes

:@*
validate_shape(
љ
save_15/Assign_9Assignpi/dense/kernel/Adam_1save_15/RestoreV2:9*
validate_shape(*
use_locking(*
_output_shapes

:@*"
_class
loc:@pi/dense/kernel*
T0
і
save_15/Assign_10Assignpi/dense_1/biassave_15/RestoreV2:10*
_output_shapes
:@*"
_class
loc:@pi/dense_1/bias*
use_locking(*
T0*
validate_shape(
є
save_15/Assign_11Assignpi/dense_1/bias/Adamsave_15/RestoreV2:11*
use_locking(*
validate_shape(*"
_class
loc:@pi/dense_1/bias*
T0*
_output_shapes
:@
ї
save_15/Assign_12Assignpi/dense_1/bias/Adam_1save_15/RestoreV2:12*
use_locking(*
_output_shapes
:@*
T0*
validate_shape(*"
_class
loc:@pi/dense_1/bias
Љ
save_15/Assign_13Assignpi/dense_1/kernelsave_15/RestoreV2:13*$
_class
loc:@pi/dense_1/kernel*
use_locking(*
T0*
_output_shapes

:@@*
validate_shape(
Ѕ
save_15/Assign_14Assignpi/dense_1/kernel/Adamsave_15/RestoreV2:14*$
_class
loc:@pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@*
use_locking(*
T0
√
save_15/Assign_15Assignpi/dense_1/kernel/Adam_1save_15/RestoreV2:15*
T0*
validate_shape(*$
_class
loc:@pi/dense_1/kernel*
use_locking(*
_output_shapes

:@@
і
save_15/Assign_16Assignpi/dense_2/biassave_15/RestoreV2:16*
use_locking(*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:*
validate_shape(*
T0
є
save_15/Assign_17Assignpi/dense_2/bias/Adamsave_15/RestoreV2:17*
validate_shape(*
_output_shapes
:*"
_class
loc:@pi/dense_2/bias*
use_locking(*
T0
ї
save_15/Assign_18Assignpi/dense_2/bias/Adam_1save_15/RestoreV2:18*
use_locking(*"
_class
loc:@pi/dense_2/bias*
T0*
validate_shape(*
_output_shapes
:
Љ
save_15/Assign_19Assignpi/dense_2/kernelsave_15/RestoreV2:19*
use_locking(*
validate_shape(*$
_class
loc:@pi/dense_2/kernel*
_output_shapes

:@*
T0
Ѕ
save_15/Assign_20Assignpi/dense_2/kernel/Adamsave_15/RestoreV2:20*
use_locking(*$
_class
loc:@pi/dense_2/kernel*
_output_shapes

:@*
T0*
validate_shape(
√
save_15/Assign_21Assignpi/dense_2/kernel/Adam_1save_15/RestoreV2:21*
use_locking(*
validate_shape(*$
_class
loc:@pi/dense_2/kernel*
T0*
_output_shapes

:@
Ѓ
save_15/Assign_22Assignv/dense/biassave_15/RestoreV2:22*
use_locking(*
_output_shapes
:@*
_class
loc:@v/dense/bias*
T0*
validate_shape(
≥
save_15/Assign_23Assignv/dense/bias/Adamsave_15/RestoreV2:23*
T0*
use_locking(*
validate_shape(*
_class
loc:@v/dense/bias*
_output_shapes
:@
µ
save_15/Assign_24Assignv/dense/bias/Adam_1save_15/RestoreV2:24*
_class
loc:@v/dense/bias*
use_locking(*
_output_shapes
:@*
validate_shape(*
T0
ґ
save_15/Assign_25Assignv/dense/kernelsave_15/RestoreV2:25*
use_locking(*
_output_shapes

:@*
validate_shape(*
T0*!
_class
loc:@v/dense/kernel
ї
save_15/Assign_26Assignv/dense/kernel/Adamsave_15/RestoreV2:26*
_output_shapes

:@*
T0*
use_locking(*!
_class
loc:@v/dense/kernel*
validate_shape(
љ
save_15/Assign_27Assignv/dense/kernel/Adam_1save_15/RestoreV2:27*
_output_shapes

:@*
use_locking(*
T0*
validate_shape(*!
_class
loc:@v/dense/kernel
≤
save_15/Assign_28Assignv/dense_1/biassave_15/RestoreV2:28*
validate_shape(*
use_locking(*!
_class
loc:@v/dense_1/bias*
_output_shapes
:@*
T0
Ј
save_15/Assign_29Assignv/dense_1/bias/Adamsave_15/RestoreV2:29*
use_locking(*
_output_shapes
:@*!
_class
loc:@v/dense_1/bias*
T0*
validate_shape(
є
save_15/Assign_30Assignv/dense_1/bias/Adam_1save_15/RestoreV2:30*
T0*!
_class
loc:@v/dense_1/bias*
_output_shapes
:@*
validate_shape(*
use_locking(
Ї
save_15/Assign_31Assignv/dense_1/kernelsave_15/RestoreV2:31*
T0*
_output_shapes

:@@*#
_class
loc:@v/dense_1/kernel*
validate_shape(*
use_locking(
њ
save_15/Assign_32Assignv/dense_1/kernel/Adamsave_15/RestoreV2:32*
use_locking(*#
_class
loc:@v/dense_1/kernel*
validate_shape(*
_output_shapes

:@@*
T0
Ѕ
save_15/Assign_33Assignv/dense_1/kernel/Adam_1save_15/RestoreV2:33*
_output_shapes

:@@*
use_locking(*#
_class
loc:@v/dense_1/kernel*
T0*
validate_shape(
≤
save_15/Assign_34Assignv/dense_2/biassave_15/RestoreV2:34*
validate_shape(*
use_locking(*!
_class
loc:@v/dense_2/bias*
T0*
_output_shapes
:
Ј
save_15/Assign_35Assignv/dense_2/bias/Adamsave_15/RestoreV2:35*
validate_shape(*
T0*!
_class
loc:@v/dense_2/bias*
use_locking(*
_output_shapes
:
є
save_15/Assign_36Assignv/dense_2/bias/Adam_1save_15/RestoreV2:36*
_output_shapes
:*
validate_shape(*
T0*
use_locking(*!
_class
loc:@v/dense_2/bias
Ї
save_15/Assign_37Assignv/dense_2/kernelsave_15/RestoreV2:37*#
_class
loc:@v/dense_2/kernel*
validate_shape(*
_output_shapes

:@*
T0*
use_locking(
њ
save_15/Assign_38Assignv/dense_2/kernel/Adamsave_15/RestoreV2:38*
use_locking(*#
_class
loc:@v/dense_2/kernel*
_output_shapes

:@*
validate_shape(*
T0
Ѕ
save_15/Assign_39Assignv/dense_2/kernel/Adam_1save_15/RestoreV2:39*
validate_shape(*
T0*
use_locking(*#
_class
loc:@v/dense_2/kernel*
_output_shapes

:@
±
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
save_16/filenamePlaceholderWithDefaultsave_16/filename/input*
shape: *
_output_shapes
: *
dtype0
k
save_16/ConstPlaceholderWithDefaultsave_16/filename*
dtype0*
_output_shapes
: *
shape: 
З
save_16/StringJoin/inputs_1Const*<
value3B1 B+_temp_c940a865352e48c8ac821c255bc92640/part*
_output_shapes
: *
dtype0
~
save_16/StringJoin
StringJoinsave_16/Constsave_16/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
T
save_16/num_shardsConst*
_output_shapes
: *
dtype0*
value	B :
_
save_16/ShardedFilename/shardConst*
dtype0*
_output_shapes
: *
value	B : 
Й
save_16/ShardedFilenameShardedFilenamesave_16/StringJoinsave_16/ShardedFilename/shardsave_16/num_shards*
_output_shapes
: 
Л
save_16/SaveV2/tensor_namesConst*ї
value±BЃ(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
_output_shapes
:(*
dtype0
ґ
save_16/SaveV2/shape_and_slicesConst*
_output_shapes
:(*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
Ќ
save_16/SaveV2SaveV2save_16/ShardedFilenamesave_16/SaveV2/tensor_namessave_16/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1v/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1v/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1v/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1v/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1v/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1v/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1*6
dtypes,
*2(
Э
save_16/control_dependencyIdentitysave_16/ShardedFilename^save_16/SaveV2*
_output_shapes
: *
T0**
_class 
loc:@save_16/ShardedFilename
¶
.save_16/MergeV2Checkpoints/checkpoint_prefixesPacksave_16/ShardedFilename^save_16/control_dependency*
N*
T0*
_output_shapes
:*

axis 
Ж
save_16/MergeV2CheckpointsMergeV2Checkpoints.save_16/MergeV2Checkpoints/checkpoint_prefixessave_16/Const*
delete_old_dirs(
Ж
save_16/IdentityIdentitysave_16/Const^save_16/MergeV2Checkpoints^save_16/control_dependency*
_output_shapes
: *
T0
О
save_16/RestoreV2/tensor_namesConst*
dtype0*
_output_shapes
:(*ї
value±BЃ(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1
є
"save_16/RestoreV2/shape_and_slicesConst*
dtype0*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:(
в
save_16/RestoreV2	RestoreV2save_16/Constsave_16/RestoreV2/tensor_names"save_16/RestoreV2/shape_and_slices*ґ
_output_shapes£
†::::::::::::::::::::::::::::::::::::::::*6
dtypes,
*2(
§
save_16/AssignAssignbeta1_powersave_16/RestoreV2*
validate_shape(*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
_output_shapes
: 
©
save_16/Assign_1Assignbeta1_power_1save_16/RestoreV2:1*
validate_shape(*
T0*
_class
loc:@v/dense/bias*
_output_shapes
: *
use_locking(
®
save_16/Assign_2Assignbeta2_powersave_16/RestoreV2:2*
_output_shapes
: * 
_class
loc:@pi/dense/bias*
use_locking(*
validate_shape(*
T0
©
save_16/Assign_3Assignbeta2_power_1save_16/RestoreV2:3*
_class
loc:@v/dense/bias*
T0*
_output_shapes
: *
use_locking(*
validate_shape(
Ѓ
save_16/Assign_4Assignpi/dense/biassave_16/RestoreV2:4* 
_class
loc:@pi/dense/bias*
use_locking(*
validate_shape(*
_output_shapes
:@*
T0
≥
save_16/Assign_5Assignpi/dense/bias/Adamsave_16/RestoreV2:5*
_output_shapes
:@*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
use_locking(
µ
save_16/Assign_6Assignpi/dense/bias/Adam_1save_16/RestoreV2:6*
T0*
use_locking(*
_output_shapes
:@*
validate_shape(* 
_class
loc:@pi/dense/bias
ґ
save_16/Assign_7Assignpi/dense/kernelsave_16/RestoreV2:7*
_output_shapes

:@*
use_locking(*
T0*
validate_shape(*"
_class
loc:@pi/dense/kernel
ї
save_16/Assign_8Assignpi/dense/kernel/Adamsave_16/RestoreV2:8*
T0*
use_locking(*"
_class
loc:@pi/dense/kernel*
_output_shapes

:@*
validate_shape(
љ
save_16/Assign_9Assignpi/dense/kernel/Adam_1save_16/RestoreV2:9*
use_locking(*
_output_shapes

:@*
T0*
validate_shape(*"
_class
loc:@pi/dense/kernel
і
save_16/Assign_10Assignpi/dense_1/biassave_16/RestoreV2:10*"
_class
loc:@pi/dense_1/bias*
use_locking(*
validate_shape(*
T0*
_output_shapes
:@
є
save_16/Assign_11Assignpi/dense_1/bias/Adamsave_16/RestoreV2:11*
validate_shape(*"
_class
loc:@pi/dense_1/bias*
T0*
_output_shapes
:@*
use_locking(
ї
save_16/Assign_12Assignpi/dense_1/bias/Adam_1save_16/RestoreV2:12*
use_locking(*"
_class
loc:@pi/dense_1/bias*
T0*
validate_shape(*
_output_shapes
:@
Љ
save_16/Assign_13Assignpi/dense_1/kernelsave_16/RestoreV2:13*
validate_shape(*
use_locking(*$
_class
loc:@pi/dense_1/kernel*
T0*
_output_shapes

:@@
Ѕ
save_16/Assign_14Assignpi/dense_1/kernel/Adamsave_16/RestoreV2:14*
validate_shape(*
use_locking(*
_output_shapes

:@@*$
_class
loc:@pi/dense_1/kernel*
T0
√
save_16/Assign_15Assignpi/dense_1/kernel/Adam_1save_16/RestoreV2:15*
_output_shapes

:@@*
use_locking(*
T0*
validate_shape(*$
_class
loc:@pi/dense_1/kernel
і
save_16/Assign_16Assignpi/dense_2/biassave_16/RestoreV2:16*"
_class
loc:@pi/dense_2/bias*
use_locking(*
T0*
validate_shape(*
_output_shapes
:
є
save_16/Assign_17Assignpi/dense_2/bias/Adamsave_16/RestoreV2:17*
_output_shapes
:*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
use_locking(*
T0
ї
save_16/Assign_18Assignpi/dense_2/bias/Adam_1save_16/RestoreV2:18*
validate_shape(*
_output_shapes
:*
T0*"
_class
loc:@pi/dense_2/bias*
use_locking(
Љ
save_16/Assign_19Assignpi/dense_2/kernelsave_16/RestoreV2:19*
T0*
use_locking(*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@
Ѕ
save_16/Assign_20Assignpi/dense_2/kernel/Adamsave_16/RestoreV2:20*
_output_shapes

:@*
validate_shape(*$
_class
loc:@pi/dense_2/kernel*
T0*
use_locking(
√
save_16/Assign_21Assignpi/dense_2/kernel/Adam_1save_16/RestoreV2:21*
_output_shapes

:@*
T0*
validate_shape(*
use_locking(*$
_class
loc:@pi/dense_2/kernel
Ѓ
save_16/Assign_22Assignv/dense/biassave_16/RestoreV2:22*
_output_shapes
:@*
_class
loc:@v/dense/bias*
T0*
use_locking(*
validate_shape(
≥
save_16/Assign_23Assignv/dense/bias/Adamsave_16/RestoreV2:23*
validate_shape(*
use_locking(*
_class
loc:@v/dense/bias*
_output_shapes
:@*
T0
µ
save_16/Assign_24Assignv/dense/bias/Adam_1save_16/RestoreV2:24*
_class
loc:@v/dense/bias*
T0*
_output_shapes
:@*
use_locking(*
validate_shape(
ґ
save_16/Assign_25Assignv/dense/kernelsave_16/RestoreV2:25*
validate_shape(*
use_locking(*
_output_shapes

:@*
T0*!
_class
loc:@v/dense/kernel
ї
save_16/Assign_26Assignv/dense/kernel/Adamsave_16/RestoreV2:26*
_output_shapes

:@*
validate_shape(*
use_locking(*
T0*!
_class
loc:@v/dense/kernel
љ
save_16/Assign_27Assignv/dense/kernel/Adam_1save_16/RestoreV2:27*
_output_shapes

:@*
T0*
validate_shape(*!
_class
loc:@v/dense/kernel*
use_locking(
≤
save_16/Assign_28Assignv/dense_1/biassave_16/RestoreV2:28*
_output_shapes
:@*
T0*!
_class
loc:@v/dense_1/bias*
validate_shape(*
use_locking(
Ј
save_16/Assign_29Assignv/dense_1/bias/Adamsave_16/RestoreV2:29*
validate_shape(*
_output_shapes
:@*
use_locking(*!
_class
loc:@v/dense_1/bias*
T0
є
save_16/Assign_30Assignv/dense_1/bias/Adam_1save_16/RestoreV2:30*
validate_shape(*!
_class
loc:@v/dense_1/bias*
T0*
use_locking(*
_output_shapes
:@
Ї
save_16/Assign_31Assignv/dense_1/kernelsave_16/RestoreV2:31*
_output_shapes

:@@*
validate_shape(*#
_class
loc:@v/dense_1/kernel*
use_locking(*
T0
њ
save_16/Assign_32Assignv/dense_1/kernel/Adamsave_16/RestoreV2:32*
validate_shape(*#
_class
loc:@v/dense_1/kernel*
_output_shapes

:@@*
use_locking(*
T0
Ѕ
save_16/Assign_33Assignv/dense_1/kernel/Adam_1save_16/RestoreV2:33*
use_locking(*#
_class
loc:@v/dense_1/kernel*
T0*
_output_shapes

:@@*
validate_shape(
≤
save_16/Assign_34Assignv/dense_2/biassave_16/RestoreV2:34*!
_class
loc:@v/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
Ј
save_16/Assign_35Assignv/dense_2/bias/Adamsave_16/RestoreV2:35*
use_locking(*
T0*
validate_shape(*!
_class
loc:@v/dense_2/bias*
_output_shapes
:
є
save_16/Assign_36Assignv/dense_2/bias/Adam_1save_16/RestoreV2:36*
use_locking(*
T0*!
_class
loc:@v/dense_2/bias*
_output_shapes
:*
validate_shape(
Ї
save_16/Assign_37Assignv/dense_2/kernelsave_16/RestoreV2:37*
use_locking(*
validate_shape(*#
_class
loc:@v/dense_2/kernel*
_output_shapes

:@*
T0
њ
save_16/Assign_38Assignv/dense_2/kernel/Adamsave_16/RestoreV2:38*
use_locking(*
validate_shape(*
T0*#
_class
loc:@v/dense_2/kernel*
_output_shapes

:@
Ѕ
save_16/Assign_39Assignv/dense_2/kernel/Adam_1save_16/RestoreV2:39*
_output_shapes

:@*
validate_shape(*
T0*
use_locking(*#
_class
loc:@v/dense_2/kernel
±
save_16/restore_shardNoOp^save_16/Assign^save_16/Assign_1^save_16/Assign_10^save_16/Assign_11^save_16/Assign_12^save_16/Assign_13^save_16/Assign_14^save_16/Assign_15^save_16/Assign_16^save_16/Assign_17^save_16/Assign_18^save_16/Assign_19^save_16/Assign_2^save_16/Assign_20^save_16/Assign_21^save_16/Assign_22^save_16/Assign_23^save_16/Assign_24^save_16/Assign_25^save_16/Assign_26^save_16/Assign_27^save_16/Assign_28^save_16/Assign_29^save_16/Assign_3^save_16/Assign_30^save_16/Assign_31^save_16/Assign_32^save_16/Assign_33^save_16/Assign_34^save_16/Assign_35^save_16/Assign_36^save_16/Assign_37^save_16/Assign_38^save_16/Assign_39^save_16/Assign_4^save_16/Assign_5^save_16/Assign_6^save_16/Assign_7^save_16/Assign_8^save_16/Assign_9
3
save_16/restore_allNoOp^save_16/restore_shard "&E
save_16/Const:0save_16/Identity:0save_16/restore_all (5 @F8"
train_op

Adam
Adam_1"е%
	variables„%‘%
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
Д
pi/dense/kernel/Adam_1:0pi/dense/kernel/Adam_1/Assignpi/dense/kernel/Adam_1/read:02*pi/dense/kernel/Adam_1/Initializer/zeros:0
t
pi/dense/bias/Adam:0pi/dense/bias/Adam/Assignpi/dense/bias/Adam/read:02&pi/dense/bias/Adam/Initializer/zeros:0
|
pi/dense/bias/Adam_1:0pi/dense/bias/Adam_1/Assignpi/dense/bias/Adam_1/read:02(pi/dense/bias/Adam_1/Initializer/zeros:0
Д
pi/dense_1/kernel/Adam:0pi/dense_1/kernel/Adam/Assignpi/dense_1/kernel/Adam/read:02*pi/dense_1/kernel/Adam/Initializer/zeros:0
М
pi/dense_1/kernel/Adam_1:0pi/dense_1/kernel/Adam_1/Assignpi/dense_1/kernel/Adam_1/read:02,pi/dense_1/kernel/Adam_1/Initializer/zeros:0
|
pi/dense_1/bias/Adam:0pi/dense_1/bias/Adam/Assignpi/dense_1/bias/Adam/read:02(pi/dense_1/bias/Adam/Initializer/zeros:0
Д
pi/dense_1/bias/Adam_1:0pi/dense_1/bias/Adam_1/Assignpi/dense_1/bias/Adam_1/read:02*pi/dense_1/bias/Adam_1/Initializer/zeros:0
Д
pi/dense_2/kernel/Adam:0pi/dense_2/kernel/Adam/Assignpi/dense_2/kernel/Adam/read:02*pi/dense_2/kernel/Adam/Initializer/zeros:0
М
pi/dense_2/kernel/Adam_1:0pi/dense_2/kernel/Adam_1/Assignpi/dense_2/kernel/Adam_1/read:02,pi/dense_2/kernel/Adam_1/Initializer/zeros:0
|
pi/dense_2/bias/Adam:0pi/dense_2/bias/Adam/Assignpi/dense_2/bias/Adam/read:02(pi/dense_2/bias/Adam/Initializer/zeros:0
Д
pi/dense_2/bias/Adam_1:0pi/dense_2/bias/Adam_1/Assignpi/dense_2/bias/Adam_1/read:02*pi/dense_2/bias/Adam_1/Initializer/zeros:0
\
beta1_power_1:0beta1_power_1/Assignbeta1_power_1/read:02beta1_power_1/initial_value:0
\
beta2_power_1:0beta2_power_1/Assignbeta2_power_1/read:02beta2_power_1/initial_value:0
x
v/dense/kernel/Adam:0v/dense/kernel/Adam/Assignv/dense/kernel/Adam/read:02'v/dense/kernel/Adam/Initializer/zeros:0
А
v/dense/kernel/Adam_1:0v/dense/kernel/Adam_1/Assignv/dense/kernel/Adam_1/read:02)v/dense/kernel/Adam_1/Initializer/zeros:0
p
v/dense/bias/Adam:0v/dense/bias/Adam/Assignv/dense/bias/Adam/read:02%v/dense/bias/Adam/Initializer/zeros:0
x
v/dense/bias/Adam_1:0v/dense/bias/Adam_1/Assignv/dense/bias/Adam_1/read:02'v/dense/bias/Adam_1/Initializer/zeros:0
А
v/dense_1/kernel/Adam:0v/dense_1/kernel/Adam/Assignv/dense_1/kernel/Adam/read:02)v/dense_1/kernel/Adam/Initializer/zeros:0
И
v/dense_1/kernel/Adam_1:0v/dense_1/kernel/Adam_1/Assignv/dense_1/kernel/Adam_1/read:02+v/dense_1/kernel/Adam_1/Initializer/zeros:0
x
v/dense_1/bias/Adam:0v/dense_1/bias/Adam/Assignv/dense_1/bias/Adam/read:02'v/dense_1/bias/Adam/Initializer/zeros:0
А
v/dense_1/bias/Adam_1:0v/dense_1/bias/Adam_1/Assignv/dense_1/bias/Adam_1/read:02)v/dense_1/bias/Adam_1/Initializer/zeros:0
А
v/dense_2/kernel/Adam:0v/dense_2/kernel/Adam/Assignv/dense_2/kernel/Adam/read:02)v/dense_2/kernel/Adam/Initializer/zeros:0
И
v/dense_2/kernel/Adam_1:0v/dense_2/kernel/Adam_1/Assignv/dense_2/kernel/Adam_1/read:02+v/dense_2/kernel/Adam_1/Initializer/zeros:0
x
v/dense_2/bias/Adam:0v/dense_2/bias/Adam/Assignv/dense_2/bias/Adam/read:02'v/dense_2/bias/Adam/Initializer/zeros:0
А
v/dense_2/bias/Adam_1:0v/dense_2/bias/Adam_1/Assignv/dense_2/bias/Adam_1/read:02)v/dense_2/bias/Adam_1/Initializer/zeros:0"ў

trainable_variablesЅ
Њ

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
v/dense_2/bias:0v/dense_2/bias/Assignv/dense_2/bias/read:02"v/dense_2/bias/Initializer/zeros:08*І
serving_defaultУ
)
x$
Placeholder:0€€€€€€€€€#
v
v/Squeeze:0€€€€€€€€€%
pi
pi/Squeeze:0	€€€€€€€€€tensorflow/serving/predict