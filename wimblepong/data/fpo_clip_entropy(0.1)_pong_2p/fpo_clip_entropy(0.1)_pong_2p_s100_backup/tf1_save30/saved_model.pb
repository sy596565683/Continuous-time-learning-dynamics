љП
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
Ttype"serve*1.14.02unknown÷µ
n
PlaceholderPlaceholder*'
_output_shapes
:€€€€€€€€€*
dtype0*
shape:€€€€€€€€€
h
Placeholder_1Placeholder*#
_output_shapes
:€€€€€€€€€*
dtype0*
shape:€€€€€€€€€
h
Placeholder_2Placeholder*
dtype0*
shape:€€€€€€€€€*#
_output_shapes
:€€€€€€€€€
h
Placeholder_3Placeholder*
dtype0*
shape:€€€€€€€€€*#
_output_shapes
:€€€€€€€€€
h
Placeholder_4Placeholder*
dtype0*
shape:€€€€€€€€€*#
_output_shapes
:€€€€€€€€€
•
0pi/dense/kernel/Initializer/random_uniform/shapeConst*"
_class
loc:@pi/dense/kernel*
_output_shapes
:*
dtype0*
valueB"   @   
Ч
.pi/dense/kernel/Initializer/random_uniform/minConst*
valueB
 *феХЊ*"
_class
loc:@pi/dense/kernel*
_output_shapes
: *
dtype0
Ч
.pi/dense/kernel/Initializer/random_uniform/maxConst*"
_class
loc:@pi/dense/kernel*
valueB
 *феХ>*
_output_shapes
: *
dtype0
о
8pi/dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform0pi/dense/kernel/Initializer/random_uniform/shape*
_output_shapes

:@*

seedd*"
_class
loc:@pi/dense/kernel*
T0*
dtype0*
seed2
Џ
.pi/dense/kernel/Initializer/random_uniform/subSub.pi/dense/kernel/Initializer/random_uniform/max.pi/dense/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@pi/dense/kernel*
_output_shapes
: 
м
.pi/dense/kernel/Initializer/random_uniform/mulMul8pi/dense/kernel/Initializer/random_uniform/RandomUniform.pi/dense/kernel/Initializer/random_uniform/sub*"
_class
loc:@pi/dense/kernel*
_output_shapes

:@*
T0
ё
*pi/dense/kernel/Initializer/random_uniformAdd.pi/dense/kernel/Initializer/random_uniform/mul.pi/dense/kernel/Initializer/random_uniform/min*
_output_shapes

:@*"
_class
loc:@pi/dense/kernel*
T0
І
pi/dense/kernel
VariableV2*
dtype0*
_output_shapes

:@*
shared_name *
	container *
shape
:@*"
_class
loc:@pi/dense/kernel
”
pi/dense/kernel/AssignAssignpi/dense/kernel*pi/dense/kernel/Initializer/random_uniform*
_output_shapes

:@*"
_class
loc:@pi/dense/kernel*
validate_shape(*
use_locking(*
T0
~
pi/dense/kernel/readIdentitypi/dense/kernel*
_output_shapes

:@*"
_class
loc:@pi/dense/kernel*
T0
О
pi/dense/bias/Initializer/zerosConst* 
_class
loc:@pi/dense/bias*
dtype0*
valueB@*    *
_output_shapes
:@
Ы
pi/dense/bias
VariableV2*
dtype0*
	container *
shape:@* 
_class
loc:@pi/dense/bias*
shared_name *
_output_shapes
:@
Њ
pi/dense/bias/AssignAssignpi/dense/biaspi/dense/bias/Initializer/zeros* 
_class
loc:@pi/dense/bias*
_output_shapes
:@*
validate_shape(*
T0*
use_locking(
t
pi/dense/bias/readIdentitypi/dense/bias* 
_class
loc:@pi/dense/bias*
_output_shapes
:@*
T0
Ф
pi/dense/MatMulMatMulPlaceholderpi/dense/kernel/read*
T0*'
_output_shapes
:€€€€€€€€€@*
transpose_a( *
transpose_b( 
Й
pi/dense/BiasAddBiasAddpi/dense/MatMulpi/dense/bias/read*
T0*'
_output_shapes
:€€€€€€€€€@*
data_formatNHWC
Y
pi/dense/TanhTanhpi/dense/BiasAdd*
T0*'
_output_shapes
:€€€€€€€€€@
©
2pi/dense_1/kernel/Initializer/random_uniform/shapeConst*
valueB"@   @   *
_output_shapes
:*$
_class
loc:@pi/dense_1/kernel*
dtype0
Ы
0pi/dense_1/kernel/Initializer/random_uniform/minConst*$
_class
loc:@pi/dense_1/kernel*
dtype0*
_output_shapes
: *
valueB
 *„≥]Њ
Ы
0pi/dense_1/kernel/Initializer/random_uniform/maxConst*
valueB
 *„≥]>*
_output_shapes
: *$
_class
loc:@pi/dense_1/kernel*
dtype0
ф
:pi/dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform2pi/dense_1/kernel/Initializer/random_uniform/shape*
_output_shapes

:@@*

seedd*
dtype0*
T0*
seed2*$
_class
loc:@pi/dense_1/kernel
в
0pi/dense_1/kernel/Initializer/random_uniform/subSub0pi/dense_1/kernel/Initializer/random_uniform/max0pi/dense_1/kernel/Initializer/random_uniform/min*
T0*
_output_shapes
: *$
_class
loc:@pi/dense_1/kernel
ф
0pi/dense_1/kernel/Initializer/random_uniform/mulMul:pi/dense_1/kernel/Initializer/random_uniform/RandomUniform0pi/dense_1/kernel/Initializer/random_uniform/sub*
_output_shapes

:@@*$
_class
loc:@pi/dense_1/kernel*
T0
ж
,pi/dense_1/kernel/Initializer/random_uniformAdd0pi/dense_1/kernel/Initializer/random_uniform/mul0pi/dense_1/kernel/Initializer/random_uniform/min*
_output_shapes

:@@*
T0*$
_class
loc:@pi/dense_1/kernel
Ђ
pi/dense_1/kernel
VariableV2*
	container *
shape
:@@*$
_class
loc:@pi/dense_1/kernel*
shared_name *
dtype0*
_output_shapes

:@@
џ
pi/dense_1/kernel/AssignAssignpi/dense_1/kernel,pi/dense_1/kernel/Initializer/random_uniform*
use_locking(*
validate_shape(*
_output_shapes

:@@*$
_class
loc:@pi/dense_1/kernel*
T0
Д
pi/dense_1/kernel/readIdentitypi/dense_1/kernel*$
_class
loc:@pi/dense_1/kernel*
_output_shapes

:@@*
T0
Т
!pi/dense_1/bias/Initializer/zerosConst*
_output_shapes
:@*"
_class
loc:@pi/dense_1/bias*
dtype0*
valueB@*    
Я
pi/dense_1/bias
VariableV2*
shape:@*
shared_name *
dtype0*"
_class
loc:@pi/dense_1/bias*
	container *
_output_shapes
:@
∆
pi/dense_1/bias/AssignAssignpi/dense_1/bias!pi/dense_1/bias/Initializer/zeros*
T0*"
_class
loc:@pi/dense_1/bias*
_output_shapes
:@*
use_locking(*
validate_shape(
z
pi/dense_1/bias/readIdentitypi/dense_1/bias*"
_class
loc:@pi/dense_1/bias*
_output_shapes
:@*
T0
Ъ
pi/dense_1/MatMulMatMulpi/dense/Tanhpi/dense_1/kernel/read*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:€€€€€€€€€@
П
pi/dense_1/BiasAddBiasAddpi/dense_1/MatMulpi/dense_1/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:€€€€€€€€€@
]
pi/dense_1/TanhTanhpi/dense_1/BiasAdd*'
_output_shapes
:€€€€€€€€€@*
T0
©
2pi/dense_2/kernel/Initializer/random_uniform/shapeConst*$
_class
loc:@pi/dense_2/kernel*
dtype0*
valueB"@      *
_output_shapes
:
Ы
0pi/dense_2/kernel/Initializer/random_uniform/minConst*
valueB
 *™7ЩЊ*$
_class
loc:@pi/dense_2/kernel*
_output_shapes
: *
dtype0
Ы
0pi/dense_2/kernel/Initializer/random_uniform/maxConst*
_output_shapes
: *$
_class
loc:@pi/dense_2/kernel*
dtype0*
valueB
 *™7Щ>
ф
:pi/dense_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform2pi/dense_2/kernel/Initializer/random_uniform/shape*$
_class
loc:@pi/dense_2/kernel*
dtype0*
T0*
_output_shapes

:@*
seed2**

seedd
в
0pi/dense_2/kernel/Initializer/random_uniform/subSub0pi/dense_2/kernel/Initializer/random_uniform/max0pi/dense_2/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*$
_class
loc:@pi/dense_2/kernel
ф
0pi/dense_2/kernel/Initializer/random_uniform/mulMul:pi/dense_2/kernel/Initializer/random_uniform/RandomUniform0pi/dense_2/kernel/Initializer/random_uniform/sub*$
_class
loc:@pi/dense_2/kernel*
T0*
_output_shapes

:@
ж
,pi/dense_2/kernel/Initializer/random_uniformAdd0pi/dense_2/kernel/Initializer/random_uniform/mul0pi/dense_2/kernel/Initializer/random_uniform/min*
_output_shapes

:@*
T0*$
_class
loc:@pi/dense_2/kernel
Ђ
pi/dense_2/kernel
VariableV2*
	container *
shape
:@*
shared_name *
dtype0*
_output_shapes

:@*$
_class
loc:@pi/dense_2/kernel
џ
pi/dense_2/kernel/AssignAssignpi/dense_2/kernel,pi/dense_2/kernel/Initializer/random_uniform*
validate_shape(*$
_class
loc:@pi/dense_2/kernel*
use_locking(*
_output_shapes

:@*
T0
Д
pi/dense_2/kernel/readIdentitypi/dense_2/kernel*$
_class
loc:@pi/dense_2/kernel*
_output_shapes

:@*
T0
Т
!pi/dense_2/bias/Initializer/zerosConst*
valueB*    *"
_class
loc:@pi/dense_2/bias*
dtype0*
_output_shapes
:
Я
pi/dense_2/bias
VariableV2*
	container *
shared_name *
_output_shapes
:*
dtype0*"
_class
loc:@pi/dense_2/bias*
shape:
∆
pi/dense_2/bias/AssignAssignpi/dense_2/bias!pi/dense_2/bias/Initializer/zeros*
validate_shape(*
use_locking(*
_output_shapes
:*"
_class
loc:@pi/dense_2/bias*
T0
z
pi/dense_2/bias/readIdentitypi/dense_2/bias*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:*
T0
Ь
pi/dense_2/MatMulMatMulpi/dense_1/Tanhpi/dense_2/kernel/read*
T0*'
_output_shapes
:€€€€€€€€€*
transpose_b( *
transpose_a( 
П
pi/dense_2/BiasAddBiasAddpi/dense_2/MatMulpi/dense_2/bias/read*'
_output_shapes
:€€€€€€€€€*
data_formatNHWC*
T0
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
ƒ
pi/multinomial/MultinomialMultinomialpi/dense_2/BiasAdd&pi/multinomial/Multinomial/num_samples*
output_dtype0	*
T0*'
_output_shapes
:€€€€€€€€€*

seedd*
seed28
v

pi/SqueezeSqueezepi/multinomial/Multinomial*
T0	*#
_output_shapes
:€€€€€€€€€*
squeeze_dims

X
pi/one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
Y
pi/one_hot/off_valueConst*
valueB
 *    *
_output_shapes
: *
dtype0
R
pi/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :
±

pi/one_hotOneHotPlaceholder_1pi/one_hot/depthpi/one_hot/on_valuepi/one_hot/off_value*
axis€€€€€€€€€*'
_output_shapes
:€€€€€€€€€*
TI0*
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
pi/SumSumpi/mulpi/Sum/reduction_indices*
	keep_dims( *
T0*

Tidx0*#
_output_shapes
:€€€€€€€€€
Z
pi/one_hot_1/on_valueConst*
valueB
 *  А?*
_output_shapes
: *
dtype0
[
pi/one_hot_1/off_valueConst*
dtype0*
_output_shapes
: *
valueB
 *    
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
axis€€€€€€€€€*'
_output_shapes
:€€€€€€€€€*
TI0	
^
pi/mul_1Mulpi/one_hot_1pi/LogSoftmax*'
_output_shapes
:€€€€€€€€€*
T0
\
pi/Sum_1/reduction_indicesConst*
value	B :*
_output_shapes
: *
dtype0
А
pi/Sum_1Sumpi/mul_1pi/Sum_1/reduction_indices*

Tidx0*
T0*
	keep_dims( *#
_output_shapes
:€€€€€€€€€
£
/v/dense/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*
valueB"   @   *!
_class
loc:@v/dense/kernel*
dtype0
Х
-v/dense/kernel/Initializer/random_uniform/minConst*
valueB
 *феХЊ*
_output_shapes
: *!
_class
loc:@v/dense/kernel*
dtype0
Х
-v/dense/kernel/Initializer/random_uniform/maxConst*!
_class
loc:@v/dense/kernel*
valueB
 *феХ>*
dtype0*
_output_shapes
: 
л
7v/dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform/v/dense/kernel/Initializer/random_uniform/shape*
T0*
_output_shapes

:@*
dtype0*!
_class
loc:@v/dense/kernel*

seedd*
seed2L
÷
-v/dense/kernel/Initializer/random_uniform/subSub-v/dense/kernel/Initializer/random_uniform/max-v/dense/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*!
_class
loc:@v/dense/kernel
и
-v/dense/kernel/Initializer/random_uniform/mulMul7v/dense/kernel/Initializer/random_uniform/RandomUniform-v/dense/kernel/Initializer/random_uniform/sub*!
_class
loc:@v/dense/kernel*
_output_shapes

:@*
T0
Џ
)v/dense/kernel/Initializer/random_uniformAdd-v/dense/kernel/Initializer/random_uniform/mul-v/dense/kernel/Initializer/random_uniform/min*
T0*
_output_shapes

:@*!
_class
loc:@v/dense/kernel
•
v/dense/kernel
VariableV2*
dtype0*!
_class
loc:@v/dense/kernel*
	container *
shape
:@*
_output_shapes

:@*
shared_name 
ѕ
v/dense/kernel/AssignAssignv/dense/kernel)v/dense/kernel/Initializer/random_uniform*
use_locking(*!
_class
loc:@v/dense/kernel*
T0*
_output_shapes

:@*
validate_shape(
{
v/dense/kernel/readIdentityv/dense/kernel*
_output_shapes

:@*
T0*!
_class
loc:@v/dense/kernel
М
v/dense/bias/Initializer/zerosConst*
_class
loc:@v/dense/bias*
_output_shapes
:@*
dtype0*
valueB@*    
Щ
v/dense/bias
VariableV2*
	container *
shared_name *
_class
loc:@v/dense/bias*
dtype0*
shape:@*
_output_shapes
:@
Ї
v/dense/bias/AssignAssignv/dense/biasv/dense/bias/Initializer/zeros*
_output_shapes
:@*
_class
loc:@v/dense/bias*
validate_shape(*
use_locking(*
T0
q
v/dense/bias/readIdentityv/dense/bias*
_output_shapes
:@*
_class
loc:@v/dense/bias*
T0
Т
v/dense/MatMulMatMulPlaceholderv/dense/kernel/read*
T0*
transpose_a( *
transpose_b( *'
_output_shapes
:€€€€€€€€€@
Ж
v/dense/BiasAddBiasAddv/dense/MatMulv/dense/bias/read*'
_output_shapes
:€€€€€€€€€@*
T0*
data_formatNHWC
W
v/dense/TanhTanhv/dense/BiasAdd*'
_output_shapes
:€€€€€€€€€@*
T0
І
1v/dense_1/kernel/Initializer/random_uniform/shapeConst*
valueB"@   @   *
dtype0*
_output_shapes
:*#
_class
loc:@v/dense_1/kernel
Щ
/v/dense_1/kernel/Initializer/random_uniform/minConst*
valueB
 *„≥]Њ*
dtype0*
_output_shapes
: *#
_class
loc:@v/dense_1/kernel
Щ
/v/dense_1/kernel/Initializer/random_uniform/maxConst*
valueB
 *„≥]>*#
_class
loc:@v/dense_1/kernel*
_output_shapes
: *
dtype0
с
9v/dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform1v/dense_1/kernel/Initializer/random_uniform/shape*
T0*

seedd*#
_class
loc:@v/dense_1/kernel*
_output_shapes

:@@*
dtype0*
seed2]
ё
/v/dense_1/kernel/Initializer/random_uniform/subSub/v/dense_1/kernel/Initializer/random_uniform/max/v/dense_1/kernel/Initializer/random_uniform/min*
T0*
_output_shapes
: *#
_class
loc:@v/dense_1/kernel
р
/v/dense_1/kernel/Initializer/random_uniform/mulMul9v/dense_1/kernel/Initializer/random_uniform/RandomUniform/v/dense_1/kernel/Initializer/random_uniform/sub*
T0*
_output_shapes

:@@*#
_class
loc:@v/dense_1/kernel
в
+v/dense_1/kernel/Initializer/random_uniformAdd/v/dense_1/kernel/Initializer/random_uniform/mul/v/dense_1/kernel/Initializer/random_uniform/min*
T0*
_output_shapes

:@@*#
_class
loc:@v/dense_1/kernel
©
v/dense_1/kernel
VariableV2*
	container *
shape
:@@*
_output_shapes

:@@*
dtype0*#
_class
loc:@v/dense_1/kernel*
shared_name 
„
v/dense_1/kernel/AssignAssignv/dense_1/kernel+v/dense_1/kernel/Initializer/random_uniform*
validate_shape(*
_output_shapes

:@@*
T0*#
_class
loc:@v/dense_1/kernel*
use_locking(
Б
v/dense_1/kernel/readIdentityv/dense_1/kernel*
T0*#
_class
loc:@v/dense_1/kernel*
_output_shapes

:@@
Р
 v/dense_1/bias/Initializer/zerosConst*
dtype0*
_output_shapes
:@*
valueB@*    *!
_class
loc:@v/dense_1/bias
Э
v/dense_1/bias
VariableV2*
shared_name *
	container *
_output_shapes
:@*
shape:@*!
_class
loc:@v/dense_1/bias*
dtype0
¬
v/dense_1/bias/AssignAssignv/dense_1/bias v/dense_1/bias/Initializer/zeros*
T0*
_output_shapes
:@*
use_locking(*!
_class
loc:@v/dense_1/bias*
validate_shape(
w
v/dense_1/bias/readIdentityv/dense_1/bias*
T0*!
_class
loc:@v/dense_1/bias*
_output_shapes
:@
Ч
v/dense_1/MatMulMatMulv/dense/Tanhv/dense_1/kernel/read*
transpose_a( *
T0*'
_output_shapes
:€€€€€€€€€@*
transpose_b( 
М
v/dense_1/BiasAddBiasAddv/dense_1/MatMulv/dense_1/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:€€€€€€€€€@
[
v/dense_1/TanhTanhv/dense_1/BiasAdd*
T0*'
_output_shapes
:€€€€€€€€€@
І
1v/dense_2/kernel/Initializer/random_uniform/shapeConst*#
_class
loc:@v/dense_2/kernel*
_output_shapes
:*
dtype0*
valueB"@      
Щ
/v/dense_2/kernel/Initializer/random_uniform/minConst*
_output_shapes
: *
valueB
 *ИОЫЊ*
dtype0*#
_class
loc:@v/dense_2/kernel
Щ
/v/dense_2/kernel/Initializer/random_uniform/maxConst*
valueB
 *ИОЫ>*
dtype0*
_output_shapes
: *#
_class
loc:@v/dense_2/kernel
с
9v/dense_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform1v/dense_2/kernel/Initializer/random_uniform/shape*
T0*#
_class
loc:@v/dense_2/kernel*
seed2n*
dtype0*
_output_shapes

:@*

seedd
ё
/v/dense_2/kernel/Initializer/random_uniform/subSub/v/dense_2/kernel/Initializer/random_uniform/max/v/dense_2/kernel/Initializer/random_uniform/min*
_output_shapes
: *#
_class
loc:@v/dense_2/kernel*
T0
р
/v/dense_2/kernel/Initializer/random_uniform/mulMul9v/dense_2/kernel/Initializer/random_uniform/RandomUniform/v/dense_2/kernel/Initializer/random_uniform/sub*
_output_shapes

:@*#
_class
loc:@v/dense_2/kernel*
T0
в
+v/dense_2/kernel/Initializer/random_uniformAdd/v/dense_2/kernel/Initializer/random_uniform/mul/v/dense_2/kernel/Initializer/random_uniform/min*#
_class
loc:@v/dense_2/kernel*
_output_shapes

:@*
T0
©
v/dense_2/kernel
VariableV2*
shape
:@*
	container *
dtype0*
_output_shapes

:@*
shared_name *#
_class
loc:@v/dense_2/kernel
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
v/dense_2/kernel/readIdentityv/dense_2/kernel*
_output_shapes

:@*#
_class
loc:@v/dense_2/kernel*
T0
Р
 v/dense_2/bias/Initializer/zerosConst*
_output_shapes
:*
dtype0*!
_class
loc:@v/dense_2/bias*
valueB*    
Э
v/dense_2/bias
VariableV2*
shape:*
	container *!
_class
loc:@v/dense_2/bias*
dtype0*
_output_shapes
:*
shared_name 
¬
v/dense_2/bias/AssignAssignv/dense_2/bias v/dense_2/bias/Initializer/zeros*
validate_shape(*
use_locking(*!
_class
loc:@v/dense_2/bias*
_output_shapes
:*
T0
w
v/dense_2/bias/readIdentityv/dense_2/bias*!
_class
loc:@v/dense_2/bias*
_output_shapes
:*
T0
Щ
v/dense_2/MatMulMatMulv/dense_1/Tanhv/dense_2/kernel/read*
T0*
transpose_a( *
transpose_b( *'
_output_shapes
:€€€€€€€€€
М
v/dense_2/BiasAddBiasAddv/dense_2/MatMulv/dense_2/bias/read*'
_output_shapes
:€€€€€€€€€*
data_formatNHWC*
T0
l
	v/SqueezeSqueezev/dense_2/BiasAdd*
T0*
squeeze_dims
*#
_output_shapes
:€€€€€€€€€
O
subSubpi/SumPlaceholder_4*#
_output_shapes
:€€€€€€€€€*
T0
=
ExpExpsub*
T0*#
_output_shapes
:€€€€€€€€€
N
	Greater/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Z
GreaterGreaterPlaceholder_2	Greater/y*
T0*#
_output_shapes
:€€€€€€€€€
J
mul/xConst*
valueB
 *ЪЩЩ?*
dtype0*
_output_shapes
: 
N
mulMulmul/xPlaceholder_2*
T0*#
_output_shapes
:€€€€€€€€€
L
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL?
R
mul_1Mulmul_1/xPlaceholder_2*
T0*#
_output_shapes
:€€€€€€€€€
S
SelectSelectGreatermulmul_1*
T0*#
_output_shapes
:€€€€€€€€€
N
mul_2MulExpPlaceholder_2*#
_output_shapes
:€€€€€€€€€*
T0
O
MinimumMinimummul_2Select*#
_output_shapes
:€€€€€€€€€*
T0
O
ConstConst*
_output_shapes
:*
valueB: *
dtype0
Z
MeanMeanMinimumConst*
	keep_dims( *
T0*
_output_shapes
: *

Tidx0
1
NegNegMean*
_output_shapes
: *
T0
T
sub_1SubPlaceholder_3	v/Squeeze*#
_output_shapes
:€€€€€€€€€*
T0
J
pow/yConst*
dtype0*
_output_shapes
: *
valueB
 *   @
F
powPowsub_1pow/y*
T0*#
_output_shapes
:€€€€€€€€€
Q
Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
Z
Mean_1MeanpowConst_1*
T0*
	keep_dims( *
_output_shapes
: *

Tidx0
Q
sub_2SubPlaceholder_4pi/Sum*#
_output_shapes
:€€€€€€€€€*
T0
Q
Const_2Const*
dtype0*
_output_shapes
:*
valueB: 
\
Mean_2Meansub_2Const_2*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
M
Neg_1Negpi/LogSoftmax*'
_output_shapes
:€€€€€€€€€*
T0
M
Exp_1Exppi/LogSoftmax*
T0*'
_output_shapes
:€€€€€€€€€
L
mul_3MulNeg_1Exp_1*'
_output_shapes
:€€€€€€€€€*
T0
X
Const_3Const*
valueB"       *
dtype0*
_output_shapes
:
\
Mean_3Meanmul_3Const_3*
_output_shapes
: *

Tidx0*
T0*
	keep_dims( 
L
mul_4/yConst*
_output_shapes
: *
valueB
 *  @@*
dtype0
>
mul_4MulMean_3mul_4/y*
_output_shapes
: *
T0
P
Greater_1/yConst*
dtype0*
_output_shapes
: *
valueB
 *ЪЩЩ?
T
	Greater_1GreaterExpGreater_1/y*
T0*#
_output_shapes
:€€€€€€€€€
K
Less/yConst*
_output_shapes
: *
valueB
 *ЌћL?*
dtype0
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

DstT0*
Truncate( *#
_output_shapes
:€€€€€€€€€*

SrcT0

Q
Const_4Const*
dtype0*
valueB: *
_output_shapes
:
[
Mean_4MeanCastConst_4*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
L
mul_5/xConst*
dtype0*
_output_shapes
: *
valueB
 *Ќћћ=
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
gradients/ShapeConst*
dtype0*
_output_shapes
: *
valueB 
X
gradients/grad_ys_0Const*
valueB
 *  А?*
dtype0*
_output_shapes
: 
o
gradients/FillFillgradients/Shapegradients/grad_ys_0*

index_type0*
_output_shapes
: *
T0
P
gradients/sub_3_grad/NegNeggradients/Fill*
T0*
_output_shapes
: 
Y
%gradients/sub_3_grad/tuple/group_depsNoOp^gradients/Fill^gradients/sub_3_grad/Neg
µ
-gradients/sub_3_grad/tuple/control_dependencyIdentitygradients/Fill&^gradients/sub_3_grad/tuple/group_deps*!
_class
loc:@gradients/Fill*
T0*
_output_shapes
: 
Ћ
/gradients/sub_3_grad/tuple/control_dependency_1Identitygradients/sub_3_grad/Neg&^gradients/sub_3_grad/tuple/group_deps*+
_class!
loc:@gradients/sub_3_grad/Neg*
_output_shapes
: *
T0
m
gradients/Neg_grad/NegNeg-gradients/sub_3_grad/tuple/control_dependency*
T0*
_output_shapes
: 
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
: *+
_class!
loc:@gradients/mul_5_grad/Mul*
T0
ѕ
/gradients/mul_5_grad/tuple/control_dependency_1Identitygradients/mul_5_grad/Mul_1&^gradients/mul_5_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/mul_5_grad/Mul_1*
_output_shapes
: 
k
!gradients/Mean_grad/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
Ф
gradients/Mean_grad/ReshapeReshapegradients/Neg_grad/Neg!gradients/Mean_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
:
`
gradients/Mean_grad/ShapeShapeMinimum*
_output_shapes
:*
out_type0*
T0
Ш
gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Shape*#
_output_shapes
:€€€€€€€€€*
T0*

Tmultiples0
b
gradients/Mean_grad/Shape_1ShapeMinimum*
T0*
_output_shapes
:*
out_type0
^
gradients/Mean_grad/Shape_2Const*
valueB *
_output_shapes
: *
dtype0
c
gradients/Mean_grad/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
Ц
gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_1gradients/Mean_grad/Const*
T0*
	keep_dims( *

Tidx0*
_output_shapes
: 
e
gradients/Mean_grad/Const_1Const*
valueB: *
_output_shapes
:*
dtype0
Ъ
gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_2gradients/Mean_grad/Const_1*
_output_shapes
: *
T0*
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
gradients/Mean_grad/floordivFloorDivgradients/Mean_grad/Prodgradients/Mean_grad/Maximum*
_output_shapes
: *
T0
~
gradients/Mean_grad/CastCastgradients/Mean_grad/floordiv*

DstT0*
Truncate( *
_output_shapes
: *

SrcT0
И
gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*
T0*#
_output_shapes
:€€€€€€€€€
z
gradients/mul_4_grad/MulMul/gradients/mul_5_grad/tuple/control_dependency_1mul_4/y*
_output_shapes
: *
T0
{
gradients/mul_4_grad/Mul_1Mul/gradients/mul_5_grad/tuple/control_dependency_1Mean_3*
T0*
_output_shapes
: 
e
%gradients/mul_4_grad/tuple/group_depsNoOp^gradients/mul_4_grad/Mul^gradients/mul_4_grad/Mul_1
…
-gradients/mul_4_grad/tuple/control_dependencyIdentitygradients/mul_4_grad/Mul&^gradients/mul_4_grad/tuple/group_deps*
_output_shapes
: *
T0*+
_class!
loc:@gradients/mul_4_grad/Mul
ѕ
/gradients/mul_4_grad/tuple/control_dependency_1Identitygradients/mul_4_grad/Mul_1&^gradients/mul_4_grad/tuple/group_deps*-
_class#
!loc:@gradients/mul_4_grad/Mul_1*
T0*
_output_shapes
: 
a
gradients/Minimum_grad/ShapeShapemul_2*
out_type0*
T0*
_output_shapes
:
d
gradients/Minimum_grad/Shape_1ShapeSelect*
T0*
out_type0*
_output_shapes
:
y
gradients/Minimum_grad/Shape_2Shapegradients/Mean_grad/truediv*
_output_shapes
:*
out_type0*
T0
g
"gradients/Minimum_grad/zeros/ConstConst*
valueB
 *    *
_output_shapes
: *
dtype0
®
gradients/Minimum_grad/zerosFillgradients/Minimum_grad/Shape_2"gradients/Minimum_grad/zeros/Const*
T0*#
_output_shapes
:€€€€€€€€€*

index_type0
j
 gradients/Minimum_grad/LessEqual	LessEqualmul_2Select*#
_output_shapes
:€€€€€€€€€*
T0
ј
,gradients/Minimum_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Minimum_grad/Shapegradients/Minimum_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0
≤
gradients/Minimum_grad/SelectSelect gradients/Minimum_grad/LessEqualgradients/Mean_grad/truedivgradients/Minimum_grad/zeros*
T0*#
_output_shapes
:€€€€€€€€€
Ѓ
gradients/Minimum_grad/SumSumgradients/Minimum_grad/Select,gradients/Minimum_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
Я
gradients/Minimum_grad/ReshapeReshapegradients/Minimum_grad/Sumgradients/Minimum_grad/Shape*
T0*#
_output_shapes
:€€€€€€€€€*
Tshape0
і
gradients/Minimum_grad/Select_1Select gradients/Minimum_grad/LessEqualgradients/Minimum_grad/zerosgradients/Mean_grad/truediv*
T0*#
_output_shapes
:€€€€€€€€€
і
gradients/Minimum_grad/Sum_1Sumgradients/Minimum_grad/Select_1.gradients/Minimum_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
T0*
	keep_dims( 
•
 gradients/Minimum_grad/Reshape_1Reshapegradients/Minimum_grad/Sum_1gradients/Minimum_grad/Shape_1*
T0*
Tshape0*#
_output_shapes
:€€€€€€€€€
s
'gradients/Minimum_grad/tuple/group_depsNoOp^gradients/Minimum_grad/Reshape!^gradients/Minimum_grad/Reshape_1
ж
/gradients/Minimum_grad/tuple/control_dependencyIdentitygradients/Minimum_grad/Reshape(^gradients/Minimum_grad/tuple/group_deps*1
_class'
%#loc:@gradients/Minimum_grad/Reshape*#
_output_shapes
:€€€€€€€€€*
T0
м
1gradients/Minimum_grad/tuple/control_dependency_1Identity gradients/Minimum_grad/Reshape_1(^gradients/Minimum_grad/tuple/group_deps*3
_class)
'%loc:@gradients/Minimum_grad/Reshape_1*
T0*#
_output_shapes
:€€€€€€€€€
t
#gradients/Mean_3_grad/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB"      
≥
gradients/Mean_3_grad/ReshapeReshape-gradients/mul_4_grad/tuple/control_dependency#gradients/Mean_3_grad/Reshape/shape*
T0*
_output_shapes

:*
Tshape0
`
gradients/Mean_3_grad/ShapeShapemul_3*
_output_shapes
:*
T0*
out_type0
Ґ
gradients/Mean_3_grad/TileTilegradients/Mean_3_grad/Reshapegradients/Mean_3_grad/Shape*
T0*'
_output_shapes
:€€€€€€€€€*

Tmultiples0
b
gradients/Mean_3_grad/Shape_1Shapemul_3*
out_type0*
T0*
_output_shapes
:
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
Ь
gradients/Mean_3_grad/ProdProdgradients/Mean_3_grad/Shape_1gradients/Mean_3_grad/Const*
	keep_dims( *
_output_shapes
: *

Tidx0*
T0
g
gradients/Mean_3_grad/Const_1Const*
_output_shapes
:*
valueB: *
dtype0
†
gradients/Mean_3_grad/Prod_1Prodgradients/Mean_3_grad/Shape_2gradients/Mean_3_grad/Const_1*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
a
gradients/Mean_3_grad/Maximum/yConst*
_output_shapes
: *
dtype0*
value	B :
И
gradients/Mean_3_grad/MaximumMaximumgradients/Mean_3_grad/Prod_1gradients/Mean_3_grad/Maximum/y*
T0*
_output_shapes
: 
Ж
gradients/Mean_3_grad/floordivFloorDivgradients/Mean_3_grad/Prodgradients/Mean_3_grad/Maximum*
_output_shapes
: *
T0
В
gradients/Mean_3_grad/CastCastgradients/Mean_3_grad/floordiv*
Truncate( *
_output_shapes
: *

DstT0*

SrcT0
Т
gradients/Mean_3_grad/truedivRealDivgradients/Mean_3_grad/Tilegradients/Mean_3_grad/Cast*'
_output_shapes
:€€€€€€€€€*
T0
]
gradients/mul_2_grad/ShapeShapeExp*
_output_shapes
:*
T0*
out_type0
i
gradients/mul_2_grad/Shape_1ShapePlaceholder_2*
out_type0*
T0*
_output_shapes
:
Ї
*gradients/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/mul_2_grad/Shapegradients/mul_2_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
Н
gradients/mul_2_grad/MulMul/gradients/Minimum_grad/tuple/control_dependencyPlaceholder_2*#
_output_shapes
:€€€€€€€€€*
T0
•
gradients/mul_2_grad/SumSumgradients/mul_2_grad/Mul*gradients/mul_2_grad/BroadcastGradientArgs*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
Щ
gradients/mul_2_grad/ReshapeReshapegradients/mul_2_grad/Sumgradients/mul_2_grad/Shape*
Tshape0*
T0*#
_output_shapes
:€€€€€€€€€
Е
gradients/mul_2_grad/Mul_1MulExp/gradients/Minimum_grad/tuple/control_dependency*
T0*#
_output_shapes
:€€€€€€€€€
Ђ
gradients/mul_2_grad/Sum_1Sumgradients/mul_2_grad/Mul_1,gradients/mul_2_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Я
gradients/mul_2_grad/Reshape_1Reshapegradients/mul_2_grad/Sum_1gradients/mul_2_grad/Shape_1*#
_output_shapes
:€€€€€€€€€*
T0*
Tshape0
m
%gradients/mul_2_grad/tuple/group_depsNoOp^gradients/mul_2_grad/Reshape^gradients/mul_2_grad/Reshape_1
ё
-gradients/mul_2_grad/tuple/control_dependencyIdentitygradients/mul_2_grad/Reshape&^gradients/mul_2_grad/tuple/group_deps*#
_output_shapes
:€€€€€€€€€*
T0*/
_class%
#!loc:@gradients/mul_2_grad/Reshape
д
/gradients/mul_2_grad/tuple/control_dependency_1Identitygradients/mul_2_grad/Reshape_1&^gradients/mul_2_grad/tuple/group_deps*1
_class'
%#loc:@gradients/mul_2_grad/Reshape_1*
T0*#
_output_shapes
:€€€€€€€€€
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
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
Э
gradients/mul_3_grad/ReshapeReshapegradients/mul_3_grad/Sumgradients/mul_3_grad/Shape*'
_output_shapes
:€€€€€€€€€*
T0*
Tshape0
y
gradients/mul_3_grad/Mul_1MulNeg_1gradients/Mean_3_grad/truediv*'
_output_shapes
:€€€€€€€€€*
T0
Ђ
gradients/mul_3_grad/Sum_1Sumgradients/mul_3_grad/Mul_1,gradients/mul_3_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
£
gradients/mul_3_grad/Reshape_1Reshapegradients/mul_3_grad/Sum_1gradients/mul_3_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:€€€€€€€€€
m
%gradients/mul_3_grad/tuple/group_depsNoOp^gradients/mul_3_grad/Reshape^gradients/mul_3_grad/Reshape_1
в
-gradients/mul_3_grad/tuple/control_dependencyIdentitygradients/mul_3_grad/Reshape&^gradients/mul_3_grad/tuple/group_deps*/
_class%
#!loc:@gradients/mul_3_grad/Reshape*'
_output_shapes
:€€€€€€€€€*
T0
и
/gradients/mul_3_grad/tuple/control_dependency_1Identitygradients/mul_3_grad/Reshape_1&^gradients/mul_3_grad/tuple/group_deps*1
_class'
%#loc:@gradients/mul_3_grad/Reshape_1*'
_output_shapes
:€€€€€€€€€*
T0

gradients/Exp_grad/mulMul-gradients/mul_2_grad/tuple/control_dependencyExp*#
_output_shapes
:€€€€€€€€€*
T0
А
gradients/Neg_1_grad/NegNeg-gradients/mul_3_grad/tuple/control_dependency*
T0*'
_output_shapes
:€€€€€€€€€
Й
gradients/Exp_1_grad/mulMul/gradients/mul_3_grad/tuple/control_dependency_1Exp_1*
T0*'
_output_shapes
:€€€€€€€€€
^
gradients/sub_grad/ShapeShapepi/Sum*
out_type0*
T0*
_output_shapes
:
g
gradients/sub_grad/Shape_1ShapePlaceholder_4*
_output_shapes
:*
out_type0*
T0
і
(gradients/sub_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/sub_grad/Shapegradients/sub_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
Я
gradients/sub_grad/SumSumgradients/Exp_grad/mul(gradients/sub_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
У
gradients/sub_grad/ReshapeReshapegradients/sub_grad/Sumgradients/sub_grad/Shape*
Tshape0*#
_output_shapes
:€€€€€€€€€*
T0
£
gradients/sub_grad/Sum_1Sumgradients/Exp_grad/mul*gradients/sub_grad/BroadcastGradientArgs:1*
	keep_dims( *
_output_shapes
:*

Tidx0*
T0
Z
gradients/sub_grad/NegNeggradients/sub_grad/Sum_1*
_output_shapes
:*
T0
Ч
gradients/sub_grad/Reshape_1Reshapegradients/sub_grad/Neggradients/sub_grad/Shape_1*#
_output_shapes
:€€€€€€€€€*
T0*
Tshape0
g
#gradients/sub_grad/tuple/group_depsNoOp^gradients/sub_grad/Reshape^gradients/sub_grad/Reshape_1
÷
+gradients/sub_grad/tuple/control_dependencyIdentitygradients/sub_grad/Reshape$^gradients/sub_grad/tuple/group_deps*#
_output_shapes
:€€€€€€€€€*-
_class#
!loc:@gradients/sub_grad/Reshape*
T0
№
-gradients/sub_grad/tuple/control_dependency_1Identitygradients/sub_grad/Reshape_1$^gradients/sub_grad/tuple/group_deps*
T0*#
_output_shapes
:€€€€€€€€€*/
_class%
#!loc:@gradients/sub_grad/Reshape_1
a
gradients/pi/Sum_grad/ShapeShapepi/mul*
_output_shapes
:*
T0*
out_type0
М
gradients/pi/Sum_grad/SizeConst*
_output_shapes
: *
value	B :*
dtype0*.
_class$
" loc:@gradients/pi/Sum_grad/Shape
І
gradients/pi/Sum_grad/addAddpi/Sum/reduction_indicesgradients/pi/Sum_grad/Size*
T0*.
_class$
" loc:@gradients/pi/Sum_grad/Shape*
_output_shapes
: 
≠
gradients/pi/Sum_grad/modFloorModgradients/pi/Sum_grad/addgradients/pi/Sum_grad/Size*.
_class$
" loc:@gradients/pi/Sum_grad/Shape*
T0*
_output_shapes
: 
Р
gradients/pi/Sum_grad/Shape_1Const*
_output_shapes
: *
dtype0*
valueB *.
_class$
" loc:@gradients/pi/Sum_grad/Shape
У
!gradients/pi/Sum_grad/range/startConst*
value	B : *.
_class$
" loc:@gradients/pi/Sum_grad/Shape*
_output_shapes
: *
dtype0
У
!gradients/pi/Sum_grad/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :*.
_class$
" loc:@gradients/pi/Sum_grad/Shape
ё
gradients/pi/Sum_grad/rangeRange!gradients/pi/Sum_grad/range/startgradients/pi/Sum_grad/Size!gradients/pi/Sum_grad/range/delta*.
_class$
" loc:@gradients/pi/Sum_grad/Shape*

Tidx0*
_output_shapes
:
Т
 gradients/pi/Sum_grad/Fill/valueConst*
value	B :*
_output_shapes
: *.
_class$
" loc:@gradients/pi/Sum_grad/Shape*
dtype0
∆
gradients/pi/Sum_grad/FillFillgradients/pi/Sum_grad/Shape_1 gradients/pi/Sum_grad/Fill/value*
T0*

index_type0*
_output_shapes
: *.
_class$
" loc:@gradients/pi/Sum_grad/Shape
Г
#gradients/pi/Sum_grad/DynamicStitchDynamicStitchgradients/pi/Sum_grad/rangegradients/pi/Sum_grad/modgradients/pi/Sum_grad/Shapegradients/pi/Sum_grad/Fill*.
_class$
" loc:@gradients/pi/Sum_grad/Shape*
T0*
N*
_output_shapes
:
С
gradients/pi/Sum_grad/Maximum/yConst*
dtype0*.
_class$
" loc:@gradients/pi/Sum_grad/Shape*
value	B :*
_output_shapes
: 
√
gradients/pi/Sum_grad/MaximumMaximum#gradients/pi/Sum_grad/DynamicStitchgradients/pi/Sum_grad/Maximum/y*
_output_shapes
:*.
_class$
" loc:@gradients/pi/Sum_grad/Shape*
T0
ї
gradients/pi/Sum_grad/floordivFloorDivgradients/pi/Sum_grad/Shapegradients/pi/Sum_grad/Maximum*
T0*.
_class$
" loc:@gradients/pi/Sum_grad/Shape*
_output_shapes
:
√
gradients/pi/Sum_grad/ReshapeReshape+gradients/sub_grad/tuple/control_dependency#gradients/pi/Sum_grad/DynamicStitch*
Tshape0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€*
T0
•
gradients/pi/Sum_grad/TileTilegradients/pi/Sum_grad/Reshapegradients/pi/Sum_grad/floordiv*
T0*'
_output_shapes
:€€€€€€€€€*

Tmultiples0
e
gradients/pi/mul_grad/ShapeShape
pi/one_hot*
T0*
_output_shapes
:*
out_type0
j
gradients/pi/mul_grad/Shape_1Shapepi/LogSoftmax*
T0*
out_type0*
_output_shapes
:
љ
+gradients/pi/mul_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/pi/mul_grad/Shapegradients/pi/mul_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
}
gradients/pi/mul_grad/MulMulgradients/pi/Sum_grad/Tilepi/LogSoftmax*
T0*'
_output_shapes
:€€€€€€€€€
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
gradients/pi/mul_grad/Sum_1Sumgradients/pi/mul_grad/Mul_1-gradients/pi/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *
T0*

Tidx0
¶
gradients/pi/mul_grad/Reshape_1Reshapegradients/pi/mul_grad/Sum_1gradients/pi/mul_grad/Shape_1*'
_output_shapes
:€€€€€€€€€*
T0*
Tshape0
p
&gradients/pi/mul_grad/tuple/group_depsNoOp^gradients/pi/mul_grad/Reshape ^gradients/pi/mul_grad/Reshape_1
ж
.gradients/pi/mul_grad/tuple/control_dependencyIdentitygradients/pi/mul_grad/Reshape'^gradients/pi/mul_grad/tuple/group_deps*
T0*'
_output_shapes
:€€€€€€€€€*0
_class&
$"loc:@gradients/pi/mul_grad/Reshape
м
0gradients/pi/mul_grad/tuple/control_dependency_1Identitygradients/pi/mul_grad/Reshape_1'^gradients/pi/mul_grad/tuple/group_deps*
T0*'
_output_shapes
:€€€€€€€€€*2
_class(
&$loc:@gradients/pi/mul_grad/Reshape_1
д
gradients/AddNAddNgradients/Neg_1_grad/Neggradients/Exp_1_grad/mul0gradients/pi/mul_grad/tuple/control_dependency_1*'
_output_shapes
:€€€€€€€€€*
N*+
_class!
loc:@gradients/Neg_1_grad/Neg*
T0
h
 gradients/pi/LogSoftmax_grad/ExpExppi/LogSoftmax*
T0*'
_output_shapes
:€€€€€€€€€
}
2gradients/pi/LogSoftmax_grad/Sum/reduction_indicesConst*
_output_shapes
: *
valueB :
€€€€€€€€€*
dtype0
Ї
 gradients/pi/LogSoftmax_grad/SumSumgradients/AddN2gradients/pi/LogSoftmax_grad/Sum/reduction_indices*
	keep_dims(*

Tidx0*
T0*'
_output_shapes
:€€€€€€€€€
Э
 gradients/pi/LogSoftmax_grad/mulMul gradients/pi/LogSoftmax_grad/Sum gradients/pi/LogSoftmax_grad/Exp*
T0*'
_output_shapes
:€€€€€€€€€
Л
 gradients/pi/LogSoftmax_grad/subSubgradients/AddN gradients/pi/LogSoftmax_grad/mul*'
_output_shapes
:€€€€€€€€€*
T0
Ъ
-gradients/pi/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad gradients/pi/LogSoftmax_grad/sub*
_output_shapes
:*
T0*
data_formatNHWC
Н
2gradients/pi/dense_2/BiasAdd_grad/tuple/group_depsNoOp!^gradients/pi/LogSoftmax_grad/sub.^gradients/pi/dense_2/BiasAdd_grad/BiasAddGrad
Д
:gradients/pi/dense_2/BiasAdd_grad/tuple/control_dependencyIdentity gradients/pi/LogSoftmax_grad/sub3^gradients/pi/dense_2/BiasAdd_grad/tuple/group_deps*3
_class)
'%loc:@gradients/pi/LogSoftmax_grad/sub*'
_output_shapes
:€€€€€€€€€*
T0
У
<gradients/pi/dense_2/BiasAdd_grad/tuple/control_dependency_1Identity-gradients/pi/dense_2/BiasAdd_grad/BiasAddGrad3^gradients/pi/dense_2/BiasAdd_grad/tuple/group_deps*
_output_shapes
:*
T0*@
_class6
42loc:@gradients/pi/dense_2/BiasAdd_grad/BiasAddGrad
Ё
'gradients/pi/dense_2/MatMul_grad/MatMulMatMul:gradients/pi/dense_2/BiasAdd_grad/tuple/control_dependencypi/dense_2/kernel/read*
T0*
transpose_b(*
transpose_a( *'
_output_shapes
:€€€€€€€€€@
ѕ
)gradients/pi/dense_2/MatMul_grad/MatMul_1MatMulpi/dense_1/Tanh:gradients/pi/dense_2/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
transpose_a(*
T0*
_output_shapes

:@
П
1gradients/pi/dense_2/MatMul_grad/tuple/group_depsNoOp(^gradients/pi/dense_2/MatMul_grad/MatMul*^gradients/pi/dense_2/MatMul_grad/MatMul_1
Р
9gradients/pi/dense_2/MatMul_grad/tuple/control_dependencyIdentity'gradients/pi/dense_2/MatMul_grad/MatMul2^gradients/pi/dense_2/MatMul_grad/tuple/group_deps*:
_class0
.,loc:@gradients/pi/dense_2/MatMul_grad/MatMul*'
_output_shapes
:€€€€€€€€€@*
T0
Н
;gradients/pi/dense_2/MatMul_grad/tuple/control_dependency_1Identity)gradients/pi/dense_2/MatMul_grad/MatMul_12^gradients/pi/dense_2/MatMul_grad/tuple/group_deps*
_output_shapes

:@*
T0*<
_class2
0.loc:@gradients/pi/dense_2/MatMul_grad/MatMul_1
±
'gradients/pi/dense_1/Tanh_grad/TanhGradTanhGradpi/dense_1/Tanh9gradients/pi/dense_2/MatMul_grad/tuple/control_dependency*
T0*'
_output_shapes
:€€€€€€€€€@
°
-gradients/pi/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad'gradients/pi/dense_1/Tanh_grad/TanhGrad*
T0*
_output_shapes
:@*
data_formatNHWC
Ф
2gradients/pi/dense_1/BiasAdd_grad/tuple/group_depsNoOp.^gradients/pi/dense_1/BiasAdd_grad/BiasAddGrad(^gradients/pi/dense_1/Tanh_grad/TanhGrad
Т
:gradients/pi/dense_1/BiasAdd_grad/tuple/control_dependencyIdentity'gradients/pi/dense_1/Tanh_grad/TanhGrad3^gradients/pi/dense_1/BiasAdd_grad/tuple/group_deps*:
_class0
.,loc:@gradients/pi/dense_1/Tanh_grad/TanhGrad*'
_output_shapes
:€€€€€€€€€@*
T0
У
<gradients/pi/dense_1/BiasAdd_grad/tuple/control_dependency_1Identity-gradients/pi/dense_1/BiasAdd_grad/BiasAddGrad3^gradients/pi/dense_1/BiasAdd_grad/tuple/group_deps*
_output_shapes
:@*@
_class6
42loc:@gradients/pi/dense_1/BiasAdd_grad/BiasAddGrad*
T0
Ё
'gradients/pi/dense_1/MatMul_grad/MatMulMatMul:gradients/pi/dense_1/BiasAdd_grad/tuple/control_dependencypi/dense_1/kernel/read*
T0*'
_output_shapes
:€€€€€€€€€@*
transpose_b(*
transpose_a( 
Ќ
)gradients/pi/dense_1/MatMul_grad/MatMul_1MatMulpi/dense/Tanh:gradients/pi/dense_1/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes

:@@*
transpose_b( *
transpose_a(
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
;gradients/pi/dense_1/MatMul_grad/tuple/control_dependency_1Identity)gradients/pi/dense_1/MatMul_grad/MatMul_12^gradients/pi/dense_1/MatMul_grad/tuple/group_deps*
T0*<
_class2
0.loc:@gradients/pi/dense_1/MatMul_grad/MatMul_1*
_output_shapes

:@@
≠
%gradients/pi/dense/Tanh_grad/TanhGradTanhGradpi/dense/Tanh9gradients/pi/dense_1/MatMul_grad/tuple/control_dependency*
T0*'
_output_shapes
:€€€€€€€€€@
Э
+gradients/pi/dense/BiasAdd_grad/BiasAddGradBiasAddGrad%gradients/pi/dense/Tanh_grad/TanhGrad*
_output_shapes
:@*
data_formatNHWC*
T0
О
0gradients/pi/dense/BiasAdd_grad/tuple/group_depsNoOp,^gradients/pi/dense/BiasAdd_grad/BiasAddGrad&^gradients/pi/dense/Tanh_grad/TanhGrad
К
8gradients/pi/dense/BiasAdd_grad/tuple/control_dependencyIdentity%gradients/pi/dense/Tanh_grad/TanhGrad1^gradients/pi/dense/BiasAdd_grad/tuple/group_deps*
T0*8
_class.
,*loc:@gradients/pi/dense/Tanh_grad/TanhGrad*'
_output_shapes
:€€€€€€€€€@
Л
:gradients/pi/dense/BiasAdd_grad/tuple/control_dependency_1Identity+gradients/pi/dense/BiasAdd_grad/BiasAddGrad1^gradients/pi/dense/BiasAdd_grad/tuple/group_deps*>
_class4
20loc:@gradients/pi/dense/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:@
„
%gradients/pi/dense/MatMul_grad/MatMulMatMul8gradients/pi/dense/BiasAdd_grad/tuple/control_dependencypi/dense/kernel/read*
transpose_b(*
T0*
transpose_a( *'
_output_shapes
:€€€€€€€€€
«
'gradients/pi/dense/MatMul_grad/MatMul_1MatMulPlaceholder8gradients/pi/dense/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes

:@*
transpose_a(
Й
/gradients/pi/dense/MatMul_grad/tuple/group_depsNoOp&^gradients/pi/dense/MatMul_grad/MatMul(^gradients/pi/dense/MatMul_grad/MatMul_1
И
7gradients/pi/dense/MatMul_grad/tuple/control_dependencyIdentity%gradients/pi/dense/MatMul_grad/MatMul0^gradients/pi/dense/MatMul_grad/tuple/group_deps*8
_class.
,*loc:@gradients/pi/dense/MatMul_grad/MatMul*
T0*'
_output_shapes
:€€€€€€€€€
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
€€€€€€€€€*
dtype0*
_output_shapes
:
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
	Reshape_1Reshape:gradients/pi/dense/BiasAdd_grad/tuple/control_dependency_1Reshape_1/shape*
_output_shapes
:@*
Tshape0*
T0
b
Reshape_2/shapeConst*
valueB:
€€€€€€€€€*
dtype0*
_output_shapes
:
Ц
	Reshape_2Reshape;gradients/pi/dense_1/MatMul_grad/tuple/control_dependency_1Reshape_2/shape*
_output_shapes	
:А *
T0*
Tshape0
b
Reshape_3/shapeConst*
valueB:
€€€€€€€€€*
dtype0*
_output_shapes
:
Ц
	Reshape_3Reshape<gradients/pi/dense_1/BiasAdd_grad/tuple/control_dependency_1Reshape_3/shape*
Tshape0*
_output_shapes
:@*
T0
b
Reshape_4/shapeConst*
dtype0*
valueB:
€€€€€€€€€*
_output_shapes
:
Ц
	Reshape_4Reshape;gradients/pi/dense_2/MatMul_grad/tuple/control_dependency_1Reshape_4/shape*
T0*
Tshape0*
_output_shapes	
:ј
b
Reshape_5/shapeConst*
dtype0*
_output_shapes
:*
valueB:
€€€€€€€€€
Ц
	Reshape_5Reshape<gradients/pi/dense_2/BiasAdd_grad/tuple/control_dependency_1Reshape_5/shape*
T0*
Tshape0*
_output_shapes
:
M
concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
Ъ
concatConcatV2Reshape	Reshape_1	Reshape_2	Reshape_3	Reshape_4	Reshape_5concat/axis*
T0*
_output_shapes	
:√%*
N*

Tidx0
g
PyFuncPyFuncconcat*
Tout
2*
token
pyfunc_0*
_output_shapes	
:√%*
Tin
2
h
Const_5Const*-
value$B""А  @      @   ј      *
dtype0*
_output_shapes
:
Q
split/split_dimConst*
value	B : *
dtype0*
_output_shapes
: 
Ф
splitSplitVPyFuncConst_5split/split_dim*
	num_split*

Tlen0*
T0*;
_output_shapes)
':А:@:А :@:ј:
`
Reshape_6/shapeConst*
dtype0*
_output_shapes
:*
valueB"   @   
c
	Reshape_6ReshapesplitReshape_6/shape*
Tshape0*
_output_shapes

:@*
T0
Y
Reshape_7/shapeConst*
valueB:@*
_output_shapes
:*
dtype0
a
	Reshape_7Reshapesplit:1Reshape_7/shape*
Tshape0*
T0*
_output_shapes
:@
`
Reshape_8/shapeConst*
valueB"@   @   *
dtype0*
_output_shapes
:
e
	Reshape_8Reshapesplit:2Reshape_8/shape*
Tshape0*
_output_shapes

:@@*
T0
Y
Reshape_9/shapeConst*
_output_shapes
:*
valueB:@*
dtype0
a
	Reshape_9Reshapesplit:3Reshape_9/shape*
Tshape0*
_output_shapes
:@*
T0
a
Reshape_10/shapeConst*
dtype0*
valueB"@      *
_output_shapes
:
g

Reshape_10Reshapesplit:4Reshape_10/shape*
T0*
_output_shapes

:@*
Tshape0
Z
Reshape_11/shapeConst*
dtype0*
valueB:*
_output_shapes
:
c

Reshape_11Reshapesplit:5Reshape_11/shape*
Tshape0*
T0*
_output_shapes
:
А
beta1_power/initial_valueConst*
dtype0* 
_class
loc:@pi/dense/bias*
valueB
 *fff?*
_output_shapes
: 
С
beta1_power
VariableV2*
	container *
shape: *
dtype0* 
_class
loc:@pi/dense/bias*
shared_name *
_output_shapes
: 
∞
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
validate_shape(*
T0* 
_class
loc:@pi/dense/bias*
_output_shapes
: *
use_locking(
l
beta1_power/readIdentitybeta1_power* 
_class
loc:@pi/dense/bias*
_output_shapes
: *
T0
А
beta2_power/initial_valueConst*
dtype0*
_output_shapes
: * 
_class
loc:@pi/dense/bias*
valueB
 *wЊ?
С
beta2_power
VariableV2*
shape: *
	container * 
_class
loc:@pi/dense/bias*
_output_shapes
: *
shared_name *
dtype0
∞
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value* 
_class
loc:@pi/dense/bias*
validate_shape(*
use_locking(*
_output_shapes
: *
T0
l
beta2_power/readIdentitybeta2_power*
_output_shapes
: *
T0* 
_class
loc:@pi/dense/bias
Я
&pi/dense/kernel/Adam/Initializer/zerosConst*
_output_shapes

:@*
dtype0*"
_class
loc:@pi/dense/kernel*
valueB@*    
ђ
pi/dense/kernel/Adam
VariableV2*
_output_shapes

:@*"
_class
loc:@pi/dense/kernel*
shared_name *
	container *
shape
:@*
dtype0
ў
pi/dense/kernel/Adam/AssignAssignpi/dense/kernel/Adam&pi/dense/kernel/Adam/Initializer/zeros*
T0*
_output_shapes

:@*
use_locking(*"
_class
loc:@pi/dense/kernel*
validate_shape(
И
pi/dense/kernel/Adam/readIdentitypi/dense/kernel/Adam*
T0*"
_class
loc:@pi/dense/kernel*
_output_shapes

:@
°
(pi/dense/kernel/Adam_1/Initializer/zerosConst*
_output_shapes

:@*
valueB@*    *"
_class
loc:@pi/dense/kernel*
dtype0
Ѓ
pi/dense/kernel/Adam_1
VariableV2*"
_class
loc:@pi/dense/kernel*
dtype0*
shared_name *
	container *
shape
:@*
_output_shapes

:@
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
pi/dense/kernel/Adam_1/readIdentitypi/dense/kernel/Adam_1*
T0*
_output_shapes

:@*"
_class
loc:@pi/dense/kernel
У
$pi/dense/bias/Adam/Initializer/zerosConst*
valueB@*    *
_output_shapes
:@* 
_class
loc:@pi/dense/bias*
dtype0
†
pi/dense/bias/Adam
VariableV2*
	container *
shape:@*
dtype0*
_output_shapes
:@* 
_class
loc:@pi/dense/bias*
shared_name 
Ќ
pi/dense/bias/Adam/AssignAssignpi/dense/bias/Adam$pi/dense/bias/Adam/Initializer/zeros*
T0*
_output_shapes
:@*
validate_shape(*
use_locking(* 
_class
loc:@pi/dense/bias
~
pi/dense/bias/Adam/readIdentitypi/dense/bias/Adam*
T0* 
_class
loc:@pi/dense/bias*
_output_shapes
:@
Х
&pi/dense/bias/Adam_1/Initializer/zerosConst*
_output_shapes
:@*
valueB@*    * 
_class
loc:@pi/dense/bias*
dtype0
Ґ
pi/dense/bias/Adam_1
VariableV2*
	container *
_output_shapes
:@*
dtype0* 
_class
loc:@pi/dense/bias*
shape:@*
shared_name 
”
pi/dense/bias/Adam_1/AssignAssignpi/dense/bias/Adam_1&pi/dense/bias/Adam_1/Initializer/zeros*
T0*
_output_shapes
:@*
use_locking(* 
_class
loc:@pi/dense/bias*
validate_shape(
В
pi/dense/bias/Adam_1/readIdentitypi/dense/bias/Adam_1*
T0*
_output_shapes
:@* 
_class
loc:@pi/dense/bias
ѓ
8pi/dense_1/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
dtype0*$
_class
loc:@pi/dense_1/kernel*
valueB"@   @   *
_output_shapes
:
Щ
.pi/dense_1/kernel/Adam/Initializer/zeros/ConstConst*
valueB
 *    *$
_class
loc:@pi/dense_1/kernel*
_output_shapes
: *
dtype0
ы
(pi/dense_1/kernel/Adam/Initializer/zerosFill8pi/dense_1/kernel/Adam/Initializer/zeros/shape_as_tensor.pi/dense_1/kernel/Adam/Initializer/zeros/Const*

index_type0*$
_class
loc:@pi/dense_1/kernel*
_output_shapes

:@@*
T0
∞
pi/dense_1/kernel/Adam
VariableV2*
_output_shapes

:@@*
shape
:@@*
	container *$
_class
loc:@pi/dense_1/kernel*
dtype0*
shared_name 
б
pi/dense_1/kernel/Adam/AssignAssignpi/dense_1/kernel/Adam(pi/dense_1/kernel/Adam/Initializer/zeros*
_output_shapes

:@@*$
_class
loc:@pi/dense_1/kernel*
validate_shape(*
use_locking(*
T0
О
pi/dense_1/kernel/Adam/readIdentitypi/dense_1/kernel/Adam*
T0*$
_class
loc:@pi/dense_1/kernel*
_output_shapes

:@@
±
:pi/dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
valueB"@   @   *$
_class
loc:@pi/dense_1/kernel*
_output_shapes
:*
dtype0
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
T0*

index_type0*$
_class
loc:@pi/dense_1/kernel*
_output_shapes

:@@
≤
pi/dense_1/kernel/Adam_1
VariableV2*
_output_shapes

:@@*$
_class
loc:@pi/dense_1/kernel*
shape
:@@*
shared_name *
	container *
dtype0
з
pi/dense_1/kernel/Adam_1/AssignAssignpi/dense_1/kernel/Adam_1*pi/dense_1/kernel/Adam_1/Initializer/zeros*$
_class
loc:@pi/dense_1/kernel*
T0*
use_locking(*
_output_shapes

:@@*
validate_shape(
Т
pi/dense_1/kernel/Adam_1/readIdentitypi/dense_1/kernel/Adam_1*
T0*$
_class
loc:@pi/dense_1/kernel*
_output_shapes

:@@
Ч
&pi/dense_1/bias/Adam/Initializer/zerosConst*"
_class
loc:@pi/dense_1/bias*
valueB@*    *
dtype0*
_output_shapes
:@
§
pi/dense_1/bias/Adam
VariableV2*
dtype0*
	container *
_output_shapes
:@*"
_class
loc:@pi/dense_1/bias*
shared_name *
shape:@
’
pi/dense_1/bias/Adam/AssignAssignpi/dense_1/bias/Adam&pi/dense_1/bias/Adam/Initializer/zeros*"
_class
loc:@pi/dense_1/bias*
T0*
validate_shape(*
_output_shapes
:@*
use_locking(
Д
pi/dense_1/bias/Adam/readIdentitypi/dense_1/bias/Adam*
T0*"
_class
loc:@pi/dense_1/bias*
_output_shapes
:@
Щ
(pi/dense_1/bias/Adam_1/Initializer/zerosConst*"
_class
loc:@pi/dense_1/bias*
_output_shapes
:@*
dtype0*
valueB@*    
¶
pi/dense_1/bias/Adam_1
VariableV2*
shared_name *
shape:@*
_output_shapes
:@*
dtype0*"
_class
loc:@pi/dense_1/bias*
	container 
џ
pi/dense_1/bias/Adam_1/AssignAssignpi/dense_1/bias/Adam_1(pi/dense_1/bias/Adam_1/Initializer/zeros*
use_locking(*"
_class
loc:@pi/dense_1/bias*
_output_shapes
:@*
validate_shape(*
T0
И
pi/dense_1/bias/Adam_1/readIdentitypi/dense_1/bias/Adam_1*"
_class
loc:@pi/dense_1/bias*
T0*
_output_shapes
:@
£
(pi/dense_2/kernel/Adam/Initializer/zerosConst*$
_class
loc:@pi/dense_2/kernel*
_output_shapes

:@*
dtype0*
valueB@*    
∞
pi/dense_2/kernel/Adam
VariableV2*$
_class
loc:@pi/dense_2/kernel*
shape
:@*
dtype0*
shared_name *
	container *
_output_shapes

:@
б
pi/dense_2/kernel/Adam/AssignAssignpi/dense_2/kernel/Adam(pi/dense_2/kernel/Adam/Initializer/zeros*
T0*
validate_shape(*
use_locking(*
_output_shapes

:@*$
_class
loc:@pi/dense_2/kernel
О
pi/dense_2/kernel/Adam/readIdentitypi/dense_2/kernel/Adam*
T0*$
_class
loc:@pi/dense_2/kernel*
_output_shapes

:@
•
*pi/dense_2/kernel/Adam_1/Initializer/zerosConst*
valueB@*    *$
_class
loc:@pi/dense_2/kernel*
_output_shapes

:@*
dtype0
≤
pi/dense_2/kernel/Adam_1
VariableV2*
shape
:@*
shared_name *
	container *$
_class
loc:@pi/dense_2/kernel*
dtype0*
_output_shapes

:@
з
pi/dense_2/kernel/Adam_1/AssignAssignpi/dense_2/kernel/Adam_1*pi/dense_2/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@
Т
pi/dense_2/kernel/Adam_1/readIdentitypi/dense_2/kernel/Adam_1*$
_class
loc:@pi/dense_2/kernel*
T0*
_output_shapes

:@
Ч
&pi/dense_2/bias/Adam/Initializer/zerosConst*
dtype0*
valueB*    *
_output_shapes
:*"
_class
loc:@pi/dense_2/bias
§
pi/dense_2/bias/Adam
VariableV2*
dtype0*
shared_name *
_output_shapes
:*
shape:*
	container *"
_class
loc:@pi/dense_2/bias
’
pi/dense_2/bias/Adam/AssignAssignpi/dense_2/bias/Adam&pi/dense_2/bias/Adam/Initializer/zeros*
validate_shape(*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:*
use_locking(*
T0
Д
pi/dense_2/bias/Adam/readIdentitypi/dense_2/bias/Adam*"
_class
loc:@pi/dense_2/bias*
T0*
_output_shapes
:
Щ
(pi/dense_2/bias/Adam_1/Initializer/zerosConst*
_output_shapes
:*
valueB*    *
dtype0*"
_class
loc:@pi/dense_2/bias
¶
pi/dense_2/bias/Adam_1
VariableV2*
dtype0*
_output_shapes
:*"
_class
loc:@pi/dense_2/bias*
shared_name *
shape:*
	container 
џ
pi/dense_2/bias/Adam_1/AssignAssignpi/dense_2/bias/Adam_1(pi/dense_2/bias/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:*
T0*"
_class
loc:@pi/dense_2/bias*
use_locking(
И
pi/dense_2/bias/Adam_1/readIdentitypi/dense_2/bias/Adam_1*
_output_shapes
:*
T0*"
_class
loc:@pi/dense_2/bias
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
: *
dtype0*
valueB
 *wЊ?
Q
Adam/epsilonConst*
dtype0*
_output_shapes
: *
valueB
 *wћ+2
ќ
%Adam/update_pi/dense/kernel/ApplyAdam	ApplyAdampi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon	Reshape_6*
use_locking( *
use_nesterov( *
_output_shapes

:@*"
_class
loc:@pi/dense/kernel*
T0
ј
#Adam/update_pi/dense/bias/ApplyAdam	ApplyAdampi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon	Reshape_7*
use_locking( *
use_nesterov( *
T0*
_output_shapes
:@* 
_class
loc:@pi/dense/bias
Ў
'Adam/update_pi/dense_1/kernel/ApplyAdam	ApplyAdampi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon	Reshape_8*
T0*
_output_shapes

:@@*$
_class
loc:@pi/dense_1/kernel*
use_locking( *
use_nesterov( 
 
%Adam/update_pi/dense_1/bias/ApplyAdam	ApplyAdampi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon	Reshape_9*
use_locking( *"
_class
loc:@pi/dense_1/bias*
_output_shapes
:@*
T0*
use_nesterov( 
ў
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
Ћ
%Adam/update_pi/dense_2/bias/ApplyAdam	ApplyAdampi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon
Reshape_11*
use_locking( *
T0*"
_class
loc:@pi/dense_2/bias*
use_nesterov( *
_output_shapes
:
в
Adam/mulMulbeta1_power/read
Adam/beta1$^Adam/update_pi/dense/bias/ApplyAdam&^Adam/update_pi/dense/kernel/ApplyAdam&^Adam/update_pi/dense_1/bias/ApplyAdam(^Adam/update_pi/dense_1/kernel/ApplyAdam&^Adam/update_pi/dense_2/bias/ApplyAdam(^Adam/update_pi/dense_2/kernel/ApplyAdam*
_output_shapes
: * 
_class
loc:@pi/dense/bias*
T0
Ш
Adam/AssignAssignbeta1_powerAdam/mul*
T0* 
_class
loc:@pi/dense/bias*
use_locking( *
_output_shapes
: *
validate_shape(
д

Adam/mul_1Mulbeta2_power/read
Adam/beta2$^Adam/update_pi/dense/bias/ApplyAdam&^Adam/update_pi/dense/kernel/ApplyAdam&^Adam/update_pi/dense_1/bias/ApplyAdam(^Adam/update_pi/dense_1/kernel/ApplyAdam&^Adam/update_pi/dense_2/bias/ApplyAdam(^Adam/update_pi/dense_2/kernel/ApplyAdam*
T0*
_output_shapes
: * 
_class
loc:@pi/dense/bias
Ь
Adam/Assign_1Assignbeta2_power
Adam/mul_1*
_output_shapes
: *
validate_shape(*
use_locking( * 
_class
loc:@pi/dense/bias*
T0
Ь
AdamNoOp^Adam/Assign^Adam/Assign_1$^Adam/update_pi/dense/bias/ApplyAdam&^Adam/update_pi/dense/kernel/ApplyAdam&^Adam/update_pi/dense_1/bias/ApplyAdam(^Adam/update_pi/dense_1/kernel/ApplyAdam&^Adam/update_pi/dense_2/bias/ApplyAdam(^Adam/update_pi/dense_2/kernel/ApplyAdam
j
Reshape_12/shapeConst^Adam*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€
q

Reshape_12Reshapepi/dense/kernel/readReshape_12/shape*
T0*
Tshape0*
_output_shapes	
:А
j
Reshape_13/shapeConst^Adam*
valueB:
€€€€€€€€€*
_output_shapes
:*
dtype0
n

Reshape_13Reshapepi/dense/bias/readReshape_13/shape*
Tshape0*
T0*
_output_shapes
:@
j
Reshape_14/shapeConst^Adam*
_output_shapes
:*
valueB:
€€€€€€€€€*
dtype0
s

Reshape_14Reshapepi/dense_1/kernel/readReshape_14/shape*
T0*
_output_shapes	
:А *
Tshape0
j
Reshape_15/shapeConst^Adam*
valueB:
€€€€€€€€€*
_output_shapes
:*
dtype0
p

Reshape_15Reshapepi/dense_1/bias/readReshape_15/shape*
_output_shapes
:@*
Tshape0*
T0
j
Reshape_16/shapeConst^Adam*
dtype0*
_output_shapes
:*
valueB:
€€€€€€€€€
s

Reshape_16Reshapepi/dense_2/kernel/readReshape_16/shape*
T0*
_output_shapes	
:ј*
Tshape0
j
Reshape_17/shapeConst^Adam*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€
p

Reshape_17Reshapepi/dense_2/bias/readReshape_17/shape*
Tshape0*
_output_shapes
:*
T0
V
concat_1/axisConst^Adam*
dtype0*
_output_shapes
: *
value	B : 
¶
concat_1ConcatV2
Reshape_12
Reshape_13
Reshape_14
Reshape_15
Reshape_16
Reshape_17concat_1/axis*
N*

Tidx0*
_output_shapes	
:√%*
T0
h
PyFunc_1PyFuncconcat_1*
token
pyfunc_1*
_output_shapes
:*
Tin
2*
Tout
2
o
Const_6Const^Adam*
_output_shapes
:*
dtype0*-
value$B""А  @      @   ј      
Z
split_1/split_dimConst^Adam*
_output_shapes
: *
value	B : *
dtype0
Л
split_1SplitVPyFunc_1Const_6split_1/split_dim*
T0*
	num_split*

Tlen0*,
_output_shapes
::::::
h
Reshape_18/shapeConst^Adam*
_output_shapes
:*
valueB"   @   *
dtype0
g

Reshape_18Reshapesplit_1Reshape_18/shape*
Tshape0*
T0*
_output_shapes

:@
a
Reshape_19/shapeConst^Adam*
_output_shapes
:*
valueB:@*
dtype0
e

Reshape_19Reshape	split_1:1Reshape_19/shape*
T0*
_output_shapes
:@*
Tshape0
h
Reshape_20/shapeConst^Adam*
_output_shapes
:*
dtype0*
valueB"@   @   
i

Reshape_20Reshape	split_1:2Reshape_20/shape*
_output_shapes

:@@*
T0*
Tshape0
a
Reshape_21/shapeConst^Adam*
_output_shapes
:*
dtype0*
valueB:@
e

Reshape_21Reshape	split_1:3Reshape_21/shape*
_output_shapes
:@*
T0*
Tshape0
h
Reshape_22/shapeConst^Adam*
dtype0*
_output_shapes
:*
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

Reshape_23Reshape	split_1:5Reshape_23/shape*
_output_shapes
:*
T0*
Tshape0
£
AssignAssignpi/dense/kernel
Reshape_18*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel
Э
Assign_1Assignpi/dense/bias
Reshape_19*
validate_shape(* 
_class
loc:@pi/dense/bias*
use_locking(*
_output_shapes
:@*
T0
©
Assign_2Assignpi/dense_1/kernel
Reshape_20*
validate_shape(*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel*
_output_shapes

:@@
°
Assign_3Assignpi/dense_1/bias
Reshape_21*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*"
_class
loc:@pi/dense_1/bias
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
Reshape_23*"
_class
loc:@pi/dense_2/bias*
use_locking(*
_output_shapes
:*
validate_shape(*
T0
Y

group_depsNoOp^Adam^Assign	^Assign_1	^Assign_2	^Assign_3	^Assign_4	^Assign_5
(
group_deps_1NoOp^Adam^group_deps
T
gradients_1/ShapeConst*
valueB *
_output_shapes
: *
dtype0
Z
gradients_1/grad_ys_0Const*
valueB
 *  А?*
_output_shapes
: *
dtype0
u
gradients_1/FillFillgradients_1/Shapegradients_1/grad_ys_0*

index_type0*
_output_shapes
: *
T0
o
%gradients_1/Mean_1_grad/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB:
Ц
gradients_1/Mean_1_grad/ReshapeReshapegradients_1/Fill%gradients_1/Mean_1_grad/Reshape/shape*
Tshape0*
_output_shapes
:*
T0
`
gradients_1/Mean_1_grad/ShapeShapepow*
T0*
_output_shapes
:*
out_type0
§
gradients_1/Mean_1_grad/TileTilegradients_1/Mean_1_grad/Reshapegradients_1/Mean_1_grad/Shape*

Tmultiples0*
T0*#
_output_shapes
:€€€€€€€€€
b
gradients_1/Mean_1_grad/Shape_1Shapepow*
_output_shapes
:*
out_type0*
T0
b
gradients_1/Mean_1_grad/Shape_2Const*
_output_shapes
: *
valueB *
dtype0
g
gradients_1/Mean_1_grad/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
Ґ
gradients_1/Mean_1_grad/ProdProdgradients_1/Mean_1_grad/Shape_1gradients_1/Mean_1_grad/Const*
	keep_dims( *
T0*
_output_shapes
: *

Tidx0
i
gradients_1/Mean_1_grad/Const_1Const*
dtype0*
valueB: *
_output_shapes
:
¶
gradients_1/Mean_1_grad/Prod_1Prodgradients_1/Mean_1_grad/Shape_2gradients_1/Mean_1_grad/Const_1*
	keep_dims( *
_output_shapes
: *

Tidx0*
T0
c
!gradients_1/Mean_1_grad/Maximum/yConst*
dtype0*
value	B :*
_output_shapes
: 
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
gradients_1/Mean_1_grad/CastCast gradients_1/Mean_1_grad/floordiv*

DstT0*

SrcT0*
Truncate( *
_output_shapes
: 
Ф
gradients_1/Mean_1_grad/truedivRealDivgradients_1/Mean_1_grad/Tilegradients_1/Mean_1_grad/Cast*
T0*#
_output_shapes
:€€€€€€€€€
_
gradients_1/pow_grad/ShapeShapesub_1*
out_type0*
_output_shapes
:*
T0
_
gradients_1/pow_grad/Shape_1Const*
dtype0*
valueB *
_output_shapes
: 
Ї
*gradients_1/pow_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/pow_grad/Shapegradients_1/pow_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0
u
gradients_1/pow_grad/mulMulgradients_1/Mean_1_grad/truedivpow/y*
T0*#
_output_shapes
:€€€€€€€€€
_
gradients_1/pow_grad/sub/yConst*
dtype0*
valueB
 *  А?*
_output_shapes
: 
c
gradients_1/pow_grad/subSubpow/ygradients_1/pow_grad/sub/y*
T0*
_output_shapes
: 
n
gradients_1/pow_grad/PowPowsub_1gradients_1/pow_grad/sub*
T0*#
_output_shapes
:€€€€€€€€€
Г
gradients_1/pow_grad/mul_1Mulgradients_1/pow_grad/mulgradients_1/pow_grad/Pow*#
_output_shapes
:€€€€€€€€€*
T0
І
gradients_1/pow_grad/SumSumgradients_1/pow_grad/mul_1*gradients_1/pow_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Щ
gradients_1/pow_grad/ReshapeReshapegradients_1/pow_grad/Sumgradients_1/pow_grad/Shape*#
_output_shapes
:€€€€€€€€€*
Tshape0*
T0
c
gradients_1/pow_grad/Greater/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
|
gradients_1/pow_grad/GreaterGreatersub_1gradients_1/pow_grad/Greater/y*
T0*#
_output_shapes
:€€€€€€€€€
i
$gradients_1/pow_grad/ones_like/ShapeShapesub_1*
T0*
out_type0*
_output_shapes
:
i
$gradients_1/pow_grad/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
≤
gradients_1/pow_grad/ones_likeFill$gradients_1/pow_grad/ones_like/Shape$gradients_1/pow_grad/ones_like/Const*#
_output_shapes
:€€€€€€€€€*
T0*

index_type0
Ш
gradients_1/pow_grad/SelectSelectgradients_1/pow_grad/Greatersub_1gradients_1/pow_grad/ones_like*
T0*#
_output_shapes
:€€€€€€€€€
j
gradients_1/pow_grad/LogLoggradients_1/pow_grad/Select*#
_output_shapes
:€€€€€€€€€*
T0
a
gradients_1/pow_grad/zeros_like	ZerosLikesub_1*
T0*#
_output_shapes
:€€€€€€€€€
Ѓ
gradients_1/pow_grad/Select_1Selectgradients_1/pow_grad/Greatergradients_1/pow_grad/Loggradients_1/pow_grad/zeros_like*
T0*#
_output_shapes
:€€€€€€€€€
u
gradients_1/pow_grad/mul_2Mulgradients_1/Mean_1_grad/truedivpow*
T0*#
_output_shapes
:€€€€€€€€€
К
gradients_1/pow_grad/mul_3Mulgradients_1/pow_grad/mul_2gradients_1/pow_grad/Select_1*#
_output_shapes
:€€€€€€€€€*
T0
Ђ
gradients_1/pow_grad/Sum_1Sumgradients_1/pow_grad/mul_3,gradients_1/pow_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
Т
gradients_1/pow_grad/Reshape_1Reshapegradients_1/pow_grad/Sum_1gradients_1/pow_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
m
%gradients_1/pow_grad/tuple/group_depsNoOp^gradients_1/pow_grad/Reshape^gradients_1/pow_grad/Reshape_1
ё
-gradients_1/pow_grad/tuple/control_dependencyIdentitygradients_1/pow_grad/Reshape&^gradients_1/pow_grad/tuple/group_deps*#
_output_shapes
:€€€€€€€€€*/
_class%
#!loc:@gradients_1/pow_grad/Reshape*
T0
„
/gradients_1/pow_grad/tuple/control_dependency_1Identitygradients_1/pow_grad/Reshape_1&^gradients_1/pow_grad/tuple/group_deps*1
_class'
%#loc:@gradients_1/pow_grad/Reshape_1*
_output_shapes
: *
T0
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
gradients_1/sub_1_grad/SumSum-gradients_1/pow_grad/tuple/control_dependency,gradients_1/sub_1_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
T0*
	keep_dims( 
Я
gradients_1/sub_1_grad/ReshapeReshapegradients_1/sub_1_grad/Sumgradients_1/sub_1_grad/Shape*#
_output_shapes
:€€€€€€€€€*
T0*
Tshape0
¬
gradients_1/sub_1_grad/Sum_1Sum-gradients_1/pow_grad/tuple/control_dependency.gradients_1/sub_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*

Tidx0*
	keep_dims( 
b
gradients_1/sub_1_grad/NegNeggradients_1/sub_1_grad/Sum_1*
_output_shapes
:*
T0
£
 gradients_1/sub_1_grad/Reshape_1Reshapegradients_1/sub_1_grad/Neggradients_1/sub_1_grad/Shape_1*
Tshape0*#
_output_shapes
:€€€€€€€€€*
T0
s
'gradients_1/sub_1_grad/tuple/group_depsNoOp^gradients_1/sub_1_grad/Reshape!^gradients_1/sub_1_grad/Reshape_1
ж
/gradients_1/sub_1_grad/tuple/control_dependencyIdentitygradients_1/sub_1_grad/Reshape(^gradients_1/sub_1_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients_1/sub_1_grad/Reshape*#
_output_shapes
:€€€€€€€€€
м
1gradients_1/sub_1_grad/tuple/control_dependency_1Identity gradients_1/sub_1_grad/Reshape_1(^gradients_1/sub_1_grad/tuple/group_deps*3
_class)
'%loc:@gradients_1/sub_1_grad/Reshape_1*#
_output_shapes
:€€€€€€€€€*
T0
q
 gradients_1/v/Squeeze_grad/ShapeShapev/dense_2/BiasAdd*
out_type0*
T0*
_output_shapes
:
¬
"gradients_1/v/Squeeze_grad/ReshapeReshape1gradients_1/sub_1_grad/tuple/control_dependency_1 gradients_1/v/Squeeze_grad/Shape*
Tshape0*
T0*'
_output_shapes
:€€€€€€€€€
Э
.gradients_1/v/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad"gradients_1/v/Squeeze_grad/Reshape*
T0*
data_formatNHWC*
_output_shapes
:
С
3gradients_1/v/dense_2/BiasAdd_grad/tuple/group_depsNoOp#^gradients_1/v/Squeeze_grad/Reshape/^gradients_1/v/dense_2/BiasAdd_grad/BiasAddGrad
К
;gradients_1/v/dense_2/BiasAdd_grad/tuple/control_dependencyIdentity"gradients_1/v/Squeeze_grad/Reshape4^gradients_1/v/dense_2/BiasAdd_grad/tuple/group_deps*5
_class+
)'loc:@gradients_1/v/Squeeze_grad/Reshape*'
_output_shapes
:€€€€€€€€€*
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
T0*
transpose_b(*
transpose_a( *'
_output_shapes
:€€€€€€€€€@
–
*gradients_1/v/dense_2/MatMul_grad/MatMul_1MatMulv/dense_1/Tanh;gradients_1/v/dense_2/BiasAdd_grad/tuple/control_dependency*
_output_shapes

:@*
transpose_b( *
T0*
transpose_a(
Т
2gradients_1/v/dense_2/MatMul_grad/tuple/group_depsNoOp)^gradients_1/v/dense_2/MatMul_grad/MatMul+^gradients_1/v/dense_2/MatMul_grad/MatMul_1
Ф
:gradients_1/v/dense_2/MatMul_grad/tuple/control_dependencyIdentity(gradients_1/v/dense_2/MatMul_grad/MatMul3^gradients_1/v/dense_2/MatMul_grad/tuple/group_deps*;
_class1
/-loc:@gradients_1/v/dense_2/MatMul_grad/MatMul*
T0*'
_output_shapes
:€€€€€€€€€@
С
<gradients_1/v/dense_2/MatMul_grad/tuple/control_dependency_1Identity*gradients_1/v/dense_2/MatMul_grad/MatMul_13^gradients_1/v/dense_2/MatMul_grad/tuple/group_deps*
_output_shapes

:@*=
_class3
1/loc:@gradients_1/v/dense_2/MatMul_grad/MatMul_1*
T0
≤
(gradients_1/v/dense_1/Tanh_grad/TanhGradTanhGradv/dense_1/Tanh:gradients_1/v/dense_2/MatMul_grad/tuple/control_dependency*
T0*'
_output_shapes
:€€€€€€€€€@
£
.gradients_1/v/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad(gradients_1/v/dense_1/Tanh_grad/TanhGrad*
_output_shapes
:@*
T0*
data_formatNHWC
Ч
3gradients_1/v/dense_1/BiasAdd_grad/tuple/group_depsNoOp/^gradients_1/v/dense_1/BiasAdd_grad/BiasAddGrad)^gradients_1/v/dense_1/Tanh_grad/TanhGrad
Ц
;gradients_1/v/dense_1/BiasAdd_grad/tuple/control_dependencyIdentity(gradients_1/v/dense_1/Tanh_grad/TanhGrad4^gradients_1/v/dense_1/BiasAdd_grad/tuple/group_deps*'
_output_shapes
:€€€€€€€€€@*
T0*;
_class1
/-loc:@gradients_1/v/dense_1/Tanh_grad/TanhGrad
Ч
=gradients_1/v/dense_1/BiasAdd_grad/tuple/control_dependency_1Identity.gradients_1/v/dense_1/BiasAdd_grad/BiasAddGrad4^gradients_1/v/dense_1/BiasAdd_grad/tuple/group_deps*
_output_shapes
:@*
T0*A
_class7
53loc:@gradients_1/v/dense_1/BiasAdd_grad/BiasAddGrad
ё
(gradients_1/v/dense_1/MatMul_grad/MatMulMatMul;gradients_1/v/dense_1/BiasAdd_grad/tuple/control_dependencyv/dense_1/kernel/read*
T0*
transpose_b(*
transpose_a( *'
_output_shapes
:€€€€€€€€€@
ќ
*gradients_1/v/dense_1/MatMul_grad/MatMul_1MatMulv/dense/Tanh;gradients_1/v/dense_1/BiasAdd_grad/tuple/control_dependency*
_output_shapes

:@@*
transpose_b( *
transpose_a(*
T0
Т
2gradients_1/v/dense_1/MatMul_grad/tuple/group_depsNoOp)^gradients_1/v/dense_1/MatMul_grad/MatMul+^gradients_1/v/dense_1/MatMul_grad/MatMul_1
Ф
:gradients_1/v/dense_1/MatMul_grad/tuple/control_dependencyIdentity(gradients_1/v/dense_1/MatMul_grad/MatMul3^gradients_1/v/dense_1/MatMul_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients_1/v/dense_1/MatMul_grad/MatMul*'
_output_shapes
:€€€€€€€€€@
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
,gradients_1/v/dense/BiasAdd_grad/BiasAddGradBiasAddGrad&gradients_1/v/dense/Tanh_grad/TanhGrad*
data_formatNHWC*
_output_shapes
:@*
T0
С
1gradients_1/v/dense/BiasAdd_grad/tuple/group_depsNoOp-^gradients_1/v/dense/BiasAdd_grad/BiasAddGrad'^gradients_1/v/dense/Tanh_grad/TanhGrad
О
9gradients_1/v/dense/BiasAdd_grad/tuple/control_dependencyIdentity&gradients_1/v/dense/Tanh_grad/TanhGrad2^gradients_1/v/dense/BiasAdd_grad/tuple/group_deps*'
_output_shapes
:€€€€€€€€€@*9
_class/
-+loc:@gradients_1/v/dense/Tanh_grad/TanhGrad*
T0
П
;gradients_1/v/dense/BiasAdd_grad/tuple/control_dependency_1Identity,gradients_1/v/dense/BiasAdd_grad/BiasAddGrad2^gradients_1/v/dense/BiasAdd_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients_1/v/dense/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@
Ў
&gradients_1/v/dense/MatMul_grad/MatMulMatMul9gradients_1/v/dense/BiasAdd_grad/tuple/control_dependencyv/dense/kernel/read*
T0*
transpose_a( *'
_output_shapes
:€€€€€€€€€*
transpose_b(
…
(gradients_1/v/dense/MatMul_grad/MatMul_1MatMulPlaceholder9gradients_1/v/dense/BiasAdd_grad/tuple/control_dependency*
_output_shapes

:@*
transpose_a(*
T0*
transpose_b( 
М
0gradients_1/v/dense/MatMul_grad/tuple/group_depsNoOp'^gradients_1/v/dense/MatMul_grad/MatMul)^gradients_1/v/dense/MatMul_grad/MatMul_1
М
8gradients_1/v/dense/MatMul_grad/tuple/control_dependencyIdentity&gradients_1/v/dense/MatMul_grad/MatMul1^gradients_1/v/dense/MatMul_grad/tuple/group_deps*9
_class/
-+loc:@gradients_1/v/dense/MatMul_grad/MatMul*
T0*'
_output_shapes
:€€€€€€€€€
Й
:gradients_1/v/dense/MatMul_grad/tuple/control_dependency_1Identity(gradients_1/v/dense/MatMul_grad/MatMul_11^gradients_1/v/dense/MatMul_grad/tuple/group_deps*;
_class1
/-loc:@gradients_1/v/dense/MatMul_grad/MatMul_1*
_output_shapes

:@*
T0
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
valueB:
€€€€€€€€€*
_output_shapes
:*
dtype0
Ч

Reshape_25Reshape;gradients_1/v/dense/BiasAdd_grad/tuple/control_dependency_1Reshape_25/shape*
Tshape0*
_output_shapes
:@*
T0
c
Reshape_26/shapeConst*
dtype0*
_output_shapes
:*
valueB:
€€€€€€€€€
Щ

Reshape_26Reshape<gradients_1/v/dense_1/MatMul_grad/tuple/control_dependency_1Reshape_26/shape*
_output_shapes	
:А *
T0*
Tshape0
c
Reshape_27/shapeConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€
Щ

Reshape_27Reshape=gradients_1/v/dense_1/BiasAdd_grad/tuple/control_dependency_1Reshape_27/shape*
Tshape0*
_output_shapes
:@*
T0
c
Reshape_28/shapeConst*
dtype0*
_output_shapes
:*
valueB:
€€€€€€€€€
Ш

Reshape_28Reshape<gradients_1/v/dense_2/MatMul_grad/tuple/control_dependency_1Reshape_28/shape*
Tshape0*
_output_shapes
:@*
T0
c
Reshape_29/shapeConst*
dtype0*
valueB:
€€€€€€€€€*
_output_shapes
:
Щ

Reshape_29Reshape=gradients_1/v/dense_2/BiasAdd_grad/tuple/control_dependency_1Reshape_29/shape*
Tshape0*
T0*
_output_shapes
:
O
concat_2/axisConst*
_output_shapes
: *
value	B : *
dtype0
¶
concat_2ConcatV2
Reshape_24
Reshape_25
Reshape_26
Reshape_27
Reshape_28
Reshape_29concat_2/axis*
N*

Tidx0*
_output_shapes	
:Ѕ$*
T0
k
PyFunc_2PyFuncconcat_2*
_output_shapes	
:Ѕ$*
Tout
2*
Tin
2*
token
pyfunc_2
h
Const_7Const*
_output_shapes
:*
dtype0*-
value$B""А  @      @   @      
S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
Щ
split_2SplitVPyFunc_2Const_7split_2/split_dim*
	num_split*
T0*:
_output_shapes(
&:А:@:А :@:@:*

Tlen0
a
Reshape_30/shapeConst*
valueB"   @   *
dtype0*
_output_shapes
:
g

Reshape_30Reshapesplit_2Reshape_30/shape*
T0*
_output_shapes

:@*
Tshape0
Z
Reshape_31/shapeConst*
_output_shapes
:*
valueB:@*
dtype0
e

Reshape_31Reshape	split_2:1Reshape_31/shape*
_output_shapes
:@*
T0*
Tshape0
a
Reshape_32/shapeConst*
valueB"@   @   *
_output_shapes
:*
dtype0
i

Reshape_32Reshape	split_2:2Reshape_32/shape*
_output_shapes

:@@*
Tshape0*
T0
Z
Reshape_33/shapeConst*
dtype0*
valueB:@*
_output_shapes
:
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
valueB"@      *
dtype0
i

Reshape_34Reshape	split_2:4Reshape_34/shape*
Tshape0*
T0*
_output_shapes

:@
Z
Reshape_35/shapeConst*
dtype0*
valueB:*
_output_shapes
:
e

Reshape_35Reshape	split_2:5Reshape_35/shape*
Tshape0*
_output_shapes
:*
T0
Б
beta1_power_1/initial_valueConst*
_output_shapes
: *
valueB
 *fff?*
dtype0*
_class
loc:@v/dense/bias
Т
beta1_power_1
VariableV2*
shape: *
_output_shapes
: *
dtype0*
_class
loc:@v/dense/bias*
	container *
shared_name 
µ
beta1_power_1/AssignAssignbeta1_power_1beta1_power_1/initial_value*
use_locking(*
_output_shapes
: *
_class
loc:@v/dense/bias*
validate_shape(*
T0
o
beta1_power_1/readIdentitybeta1_power_1*
T0*
_class
loc:@v/dense/bias*
_output_shapes
: 
Б
beta2_power_1/initial_valueConst*
_output_shapes
: *
dtype0*
valueB
 *wЊ?*
_class
loc:@v/dense/bias
Т
beta2_power_1
VariableV2*
shape: *
_class
loc:@v/dense/bias*
shared_name *
dtype0*
	container *
_output_shapes
: 
µ
beta2_power_1/AssignAssignbeta2_power_1beta2_power_1/initial_value*
validate_shape(*
use_locking(*
T0*
_class
loc:@v/dense/bias*
_output_shapes
: 
o
beta2_power_1/readIdentitybeta2_power_1*
T0*
_output_shapes
: *
_class
loc:@v/dense/bias
Э
%v/dense/kernel/Adam/Initializer/zerosConst*
dtype0*
_output_shapes

:@*!
_class
loc:@v/dense/kernel*
valueB@*    
™
v/dense/kernel/Adam
VariableV2*!
_class
loc:@v/dense/kernel*
	container *
_output_shapes

:@*
dtype0*
shared_name *
shape
:@
’
v/dense/kernel/Adam/AssignAssignv/dense/kernel/Adam%v/dense/kernel/Adam/Initializer/zeros*
_output_shapes

:@*
use_locking(*!
_class
loc:@v/dense/kernel*
T0*
validate_shape(
Е
v/dense/kernel/Adam/readIdentityv/dense/kernel/Adam*
T0*!
_class
loc:@v/dense/kernel*
_output_shapes

:@
Я
'v/dense/kernel/Adam_1/Initializer/zerosConst*
_output_shapes

:@*
dtype0*!
_class
loc:@v/dense/kernel*
valueB@*    
ђ
v/dense/kernel/Adam_1
VariableV2*
	container *
dtype0*
shape
:@*
shared_name *
_output_shapes

:@*!
_class
loc:@v/dense/kernel
џ
v/dense/kernel/Adam_1/AssignAssignv/dense/kernel/Adam_1'v/dense/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*
_output_shapes

:@*
validate_shape(*!
_class
loc:@v/dense/kernel
Й
v/dense/kernel/Adam_1/readIdentityv/dense/kernel/Adam_1*
_output_shapes

:@*
T0*!
_class
loc:@v/dense/kernel
С
#v/dense/bias/Adam/Initializer/zerosConst*
dtype0*
valueB@*    *
_output_shapes
:@*
_class
loc:@v/dense/bias
Ю
v/dense/bias/Adam
VariableV2*
_class
loc:@v/dense/bias*
	container *
shared_name *
dtype0*
shape:@*
_output_shapes
:@
…
v/dense/bias/Adam/AssignAssignv/dense/bias/Adam#v/dense/bias/Adam/Initializer/zeros*
validate_shape(*
use_locking(*
T0*
_output_shapes
:@*
_class
loc:@v/dense/bias
{
v/dense/bias/Adam/readIdentityv/dense/bias/Adam*
T0*
_output_shapes
:@*
_class
loc:@v/dense/bias
У
%v/dense/bias/Adam_1/Initializer/zerosConst*
valueB@*    *
dtype0*
_output_shapes
:@*
_class
loc:@v/dense/bias
†
v/dense/bias/Adam_1
VariableV2*
shared_name *
_class
loc:@v/dense/bias*
_output_shapes
:@*
shape:@*
	container *
dtype0
ѕ
v/dense/bias/Adam_1/AssignAssignv/dense/bias/Adam_1%v/dense/bias/Adam_1/Initializer/zeros*
_output_shapes
:@*
use_locking(*
validate_shape(*
T0*
_class
loc:@v/dense/bias

v/dense/bias/Adam_1/readIdentityv/dense/bias/Adam_1*
_output_shapes
:@*
_class
loc:@v/dense/bias*
T0
≠
7v/dense_1/kernel/Adam/Initializer/zeros/shape_as_tensorConst*#
_class
loc:@v/dense_1/kernel*
dtype0*
valueB"@   @   *
_output_shapes
:
Ч
-v/dense_1/kernel/Adam/Initializer/zeros/ConstConst*
_output_shapes
: *
dtype0*#
_class
loc:@v/dense_1/kernel*
valueB
 *    
ч
'v/dense_1/kernel/Adam/Initializer/zerosFill7v/dense_1/kernel/Adam/Initializer/zeros/shape_as_tensor-v/dense_1/kernel/Adam/Initializer/zeros/Const*
T0*
_output_shapes

:@@*

index_type0*#
_class
loc:@v/dense_1/kernel
Ѓ
v/dense_1/kernel/Adam
VariableV2*#
_class
loc:@v/dense_1/kernel*
shared_name *
_output_shapes

:@@*
dtype0*
shape
:@@*
	container 
Ё
v/dense_1/kernel/Adam/AssignAssignv/dense_1/kernel/Adam'v/dense_1/kernel/Adam/Initializer/zeros*
validate_shape(*#
_class
loc:@v/dense_1/kernel*
_output_shapes

:@@*
use_locking(*
T0
Л
v/dense_1/kernel/Adam/readIdentityv/dense_1/kernel/Adam*
T0*#
_class
loc:@v/dense_1/kernel*
_output_shapes

:@@
ѓ
9v/dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
dtype0*#
_class
loc:@v/dense_1/kernel*
valueB"@   @   *
_output_shapes
:
Щ
/v/dense_1/kernel/Adam_1/Initializer/zeros/ConstConst*
_output_shapes
: *#
_class
loc:@v/dense_1/kernel*
valueB
 *    *
dtype0
э
)v/dense_1/kernel/Adam_1/Initializer/zerosFill9v/dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensor/v/dense_1/kernel/Adam_1/Initializer/zeros/Const*
_output_shapes

:@@*
T0*

index_type0*#
_class
loc:@v/dense_1/kernel
∞
v/dense_1/kernel/Adam_1
VariableV2*
shared_name *
_output_shapes

:@@*
dtype0*
	container *#
_class
loc:@v/dense_1/kernel*
shape
:@@
г
v/dense_1/kernel/Adam_1/AssignAssignv/dense_1/kernel/Adam_1)v/dense_1/kernel/Adam_1/Initializer/zeros*
validate_shape(*#
_class
loc:@v/dense_1/kernel*
use_locking(*
_output_shapes

:@@*
T0
П
v/dense_1/kernel/Adam_1/readIdentityv/dense_1/kernel/Adam_1*#
_class
loc:@v/dense_1/kernel*
_output_shapes

:@@*
T0
Х
%v/dense_1/bias/Adam/Initializer/zerosConst*!
_class
loc:@v/dense_1/bias*
dtype0*
_output_shapes
:@*
valueB@*    
Ґ
v/dense_1/bias/Adam
VariableV2*
shared_name *
shape:@*
	container *!
_class
loc:@v/dense_1/bias*
_output_shapes
:@*
dtype0
—
v/dense_1/bias/Adam/AssignAssignv/dense_1/bias/Adam%v/dense_1/bias/Adam/Initializer/zeros*
_output_shapes
:@*
validate_shape(*
T0*!
_class
loc:@v/dense_1/bias*
use_locking(
Б
v/dense_1/bias/Adam/readIdentityv/dense_1/bias/Adam*
T0*
_output_shapes
:@*!
_class
loc:@v/dense_1/bias
Ч
'v/dense_1/bias/Adam_1/Initializer/zerosConst*!
_class
loc:@v/dense_1/bias*
valueB@*    *
_output_shapes
:@*
dtype0
§
v/dense_1/bias/Adam_1
VariableV2*
shape:@*
	container *
shared_name *
dtype0*!
_class
loc:@v/dense_1/bias*
_output_shapes
:@
„
v/dense_1/bias/Adam_1/AssignAssignv/dense_1/bias/Adam_1'v/dense_1/bias/Adam_1/Initializer/zeros*
use_locking(*!
_class
loc:@v/dense_1/bias*
_output_shapes
:@*
T0*
validate_shape(
Е
v/dense_1/bias/Adam_1/readIdentityv/dense_1/bias/Adam_1*!
_class
loc:@v/dense_1/bias*
_output_shapes
:@*
T0
°
'v/dense_2/kernel/Adam/Initializer/zerosConst*
dtype0*
_output_shapes

:@*#
_class
loc:@v/dense_2/kernel*
valueB@*    
Ѓ
v/dense_2/kernel/Adam
VariableV2*
_output_shapes

:@*
shape
:@*
	container *#
_class
loc:@v/dense_2/kernel*
dtype0*
shared_name 
Ё
v/dense_2/kernel/Adam/AssignAssignv/dense_2/kernel/Adam'v/dense_2/kernel/Adam/Initializer/zeros*
_output_shapes

:@*
use_locking(*
T0*#
_class
loc:@v/dense_2/kernel*
validate_shape(
Л
v/dense_2/kernel/Adam/readIdentityv/dense_2/kernel/Adam*
T0*#
_class
loc:@v/dense_2/kernel*
_output_shapes

:@
£
)v/dense_2/kernel/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes

:@*
valueB@*    *#
_class
loc:@v/dense_2/kernel
∞
v/dense_2/kernel/Adam_1
VariableV2*
shared_name *
_output_shapes

:@*
dtype0*#
_class
loc:@v/dense_2/kernel*
	container *
shape
:@
г
v/dense_2/kernel/Adam_1/AssignAssignv/dense_2/kernel/Adam_1)v/dense_2/kernel/Adam_1/Initializer/zeros*#
_class
loc:@v/dense_2/kernel*
_output_shapes

:@*
validate_shape(*
use_locking(*
T0
П
v/dense_2/kernel/Adam_1/readIdentityv/dense_2/kernel/Adam_1*
_output_shapes

:@*
T0*#
_class
loc:@v/dense_2/kernel
Х
%v/dense_2/bias/Adam/Initializer/zerosConst*
dtype0*!
_class
loc:@v/dense_2/bias*
_output_shapes
:*
valueB*    
Ґ
v/dense_2/bias/Adam
VariableV2*
_output_shapes
:*
shape:*
dtype0*
	container *!
_class
loc:@v/dense_2/bias*
shared_name 
—
v/dense_2/bias/Adam/AssignAssignv/dense_2/bias/Adam%v/dense_2/bias/Adam/Initializer/zeros*
_output_shapes
:*
T0*!
_class
loc:@v/dense_2/bias*
use_locking(*
validate_shape(
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
VariableV2*!
_class
loc:@v/dense_2/bias*
_output_shapes
:*
	container *
dtype0*
shape:*
shared_name 
„
v/dense_2/bias/Adam_1/AssignAssignv/dense_2/bias/Adam_1'v/dense_2/bias/Adam_1/Initializer/zeros*
validate_shape(*!
_class
loc:@v/dense_2/bias*
T0*
_output_shapes
:*
use_locking(
Е
v/dense_2/bias/Adam_1/readIdentityv/dense_2/bias/Adam_1*
T0*
_output_shapes
:*!
_class
loc:@v/dense_2/bias
Y
Adam_1/learning_rateConst*
valueB
 *oГ:*
_output_shapes
: *
dtype0
Q
Adam_1/beta1Const*
_output_shapes
: *
valueB
 *fff?*
dtype0
Q
Adam_1/beta2Const*
_output_shapes
: *
dtype0*
valueB
 *wЊ?
S
Adam_1/epsilonConst*
_output_shapes
: *
valueB
 *wћ+2*
dtype0
Ў
&Adam_1/update_v/dense/kernel/ApplyAdam	ApplyAdamv/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon
Reshape_30*
T0*!
_class
loc:@v/dense/kernel*
_output_shapes

:@*
use_nesterov( *
use_locking( 
 
$Adam_1/update_v/dense/bias/ApplyAdam	ApplyAdamv/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon
Reshape_31*
use_nesterov( *
_class
loc:@v/dense/bias*
use_locking( *
T0*
_output_shapes
:@
в
(Adam_1/update_v/dense_1/kernel/ApplyAdam	ApplyAdamv/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon
Reshape_32*
use_locking( *
_output_shapes

:@@*
T0*
use_nesterov( *#
_class
loc:@v/dense_1/kernel
‘
&Adam_1/update_v/dense_1/bias/ApplyAdam	ApplyAdamv/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon
Reshape_33*!
_class
loc:@v/dense_1/bias*
use_locking( *
T0*
use_nesterov( *
_output_shapes
:@
в
(Adam_1/update_v/dense_2/kernel/ApplyAdam	ApplyAdamv/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon
Reshape_34*
T0*#
_class
loc:@v/dense_2/kernel*
use_locking( *
use_nesterov( *
_output_shapes

:@
‘
&Adam_1/update_v/dense_2/bias/ApplyAdam	ApplyAdamv/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon
Reshape_35*
use_nesterov( *
_output_shapes
:*
T0*
use_locking( *!
_class
loc:@v/dense_2/bias
н

Adam_1/mulMulbeta1_power_1/readAdam_1/beta1%^Adam_1/update_v/dense/bias/ApplyAdam'^Adam_1/update_v/dense/kernel/ApplyAdam'^Adam_1/update_v/dense_1/bias/ApplyAdam)^Adam_1/update_v/dense_1/kernel/ApplyAdam'^Adam_1/update_v/dense_2/bias/ApplyAdam)^Adam_1/update_v/dense_2/kernel/ApplyAdam*
T0*
_output_shapes
: *
_class
loc:@v/dense/bias
Э
Adam_1/AssignAssignbeta1_power_1
Adam_1/mul*
_class
loc:@v/dense/bias*
validate_shape(*
T0*
use_locking( *
_output_shapes
: 
п
Adam_1/mul_1Mulbeta2_power_1/readAdam_1/beta2%^Adam_1/update_v/dense/bias/ApplyAdam'^Adam_1/update_v/dense/kernel/ApplyAdam'^Adam_1/update_v/dense_1/bias/ApplyAdam)^Adam_1/update_v/dense_1/kernel/ApplyAdam'^Adam_1/update_v/dense_2/bias/ApplyAdam)^Adam_1/update_v/dense_2/kernel/ApplyAdam*
_output_shapes
: *
_class
loc:@v/dense/bias*
T0
°
Adam_1/Assign_1Assignbeta2_power_1Adam_1/mul_1*
T0*
use_locking( *
_output_shapes
: *
validate_shape(*
_class
loc:@v/dense/bias
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

Reshape_36Reshapev/dense/kernel/readReshape_36/shape*
T0*
_output_shapes	
:А*
Tshape0
l
Reshape_37/shapeConst^Adam_1*
_output_shapes
:*
valueB:
€€€€€€€€€*
dtype0
m

Reshape_37Reshapev/dense/bias/readReshape_37/shape*
Tshape0*
_output_shapes
:@*
T0
l
Reshape_38/shapeConst^Adam_1*
valueB:
€€€€€€€€€*
_output_shapes
:*
dtype0
r

Reshape_38Reshapev/dense_1/kernel/readReshape_38/shape*
T0*
_output_shapes	
:А *
Tshape0
l
Reshape_39/shapeConst^Adam_1*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€
o

Reshape_39Reshapev/dense_1/bias/readReshape_39/shape*
_output_shapes
:@*
Tshape0*
T0
l
Reshape_40/shapeConst^Adam_1*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€
q

Reshape_40Reshapev/dense_2/kernel/readReshape_40/shape*
T0*
_output_shapes
:@*
Tshape0
l
Reshape_41/shapeConst^Adam_1*
dtype0*
valueB:
€€€€€€€€€*
_output_shapes
:
o

Reshape_41Reshapev/dense_2/bias/readReshape_41/shape*
_output_shapes
:*
Tshape0*
T0
X
concat_3/axisConst^Adam_1*
_output_shapes
: *
dtype0*
value	B : 
¶
concat_3ConcatV2
Reshape_36
Reshape_37
Reshape_38
Reshape_39
Reshape_40
Reshape_41concat_3/axis*
_output_shapes	
:Ѕ$*
T0*

Tidx0*
N
h
PyFunc_3PyFuncconcat_3*
Tin
2*
_output_shapes
:*
Tout
2*
token
pyfunc_3
q
Const_8Const^Adam_1*
dtype0*
_output_shapes
:*-
value$B""А  @      @   @      
\
split_3/split_dimConst^Adam_1*
_output_shapes
: *
dtype0*
value	B : 
Л
split_3SplitVPyFunc_3Const_8split_3/split_dim*
	num_split*

Tlen0*
T0*,
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

:@*
T0*
Tshape0
c
Reshape_43/shapeConst^Adam_1*
dtype0*
valueB:@*
_output_shapes
:
e

Reshape_43Reshape	split_3:1Reshape_43/shape*
_output_shapes
:@*
T0*
Tshape0
j
Reshape_44/shapeConst^Adam_1*
_output_shapes
:*
valueB"@   @   *
dtype0
i

Reshape_44Reshape	split_3:2Reshape_44/shape*
T0*
Tshape0*
_output_shapes

:@@
c
Reshape_45/shapeConst^Adam_1*
valueB:@*
_output_shapes
:*
dtype0
e

Reshape_45Reshape	split_3:3Reshape_45/shape*
Tshape0*
_output_shapes
:@*
T0
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
Reshape_47/shapeConst^Adam_1*
dtype0*
valueB:*
_output_shapes
:
e

Reshape_47Reshape	split_3:5Reshape_47/shape*
T0*
Tshape0*
_output_shapes
:
£
Assign_6Assignv/dense/kernel
Reshape_42*
use_locking(*
_output_shapes

:@*!
_class
loc:@v/dense/kernel*
validate_shape(*
T0
Ы
Assign_7Assignv/dense/bias
Reshape_43*
T0*
validate_shape(*
_class
loc:@v/dense/bias*
use_locking(*
_output_shapes
:@
І
Assign_8Assignv/dense_1/kernel
Reshape_44*
_output_shapes

:@@*
T0*
use_locking(*
validate_shape(*#
_class
loc:@v/dense_1/kernel
Я
Assign_9Assignv/dense_1/bias
Reshape_45*
use_locking(*!
_class
loc:@v/dense_1/bias*
validate_shape(*
_output_shapes
:@*
T0
®
	Assign_10Assignv/dense_2/kernel
Reshape_46*
T0*
validate_shape(*
_output_shapes

:@*#
_class
loc:@v/dense_2/kernel*
use_locking(
†
	Assign_11Assignv/dense_2/bias
Reshape_47*!
_class
loc:@v/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
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
valueB:
€€€€€€€€€*
dtype0*
_output_shapes
:
q

Reshape_48Reshapepi/dense/kernel/readReshape_48/shape*
Tshape0*
T0*
_output_shapes	
:А
c
Reshape_49/shapeConst*
valueB:
€€€€€€€€€*
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
Reshape_50/shapeConst*
valueB:
€€€€€€€€€*
_output_shapes
:*
dtype0
s

Reshape_50Reshapepi/dense_1/kernel/readReshape_50/shape*
Tshape0*
_output_shapes	
:А *
T0
c
Reshape_51/shapeConst*
dtype0*
_output_shapes
:*
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
dtype0*
_output_shapes
:*
valueB:
€€€€€€€€€
s

Reshape_52Reshapepi/dense_2/kernel/readReshape_52/shape*
_output_shapes	
:ј*
Tshape0*
T0
c
Reshape_53/shapeConst*
_output_shapes
:*
valueB:
€€€€€€€€€*
dtype0
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
€€€€€€€€€
p

Reshape_54Reshapev/dense/kernel/readReshape_54/shape*
Tshape0*
T0*
_output_shapes	
:А
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
dtype0*
valueB:
€€€€€€€€€*
_output_shapes
:
r

Reshape_56Reshapev/dense_1/kernel/readReshape_56/shape*
T0*
_output_shapes	
:А *
Tshape0
c
Reshape_57/shapeConst*
_output_shapes
:*
valueB:
€€€€€€€€€*
dtype0
o

Reshape_57Reshapev/dense_1/bias/readReshape_57/shape*
T0*
_output_shapes
:@*
Tshape0
c
Reshape_58/shapeConst*
valueB:
€€€€€€€€€*
_output_shapes
:*
dtype0
q

Reshape_58Reshapev/dense_2/kernel/readReshape_58/shape*
T0*
Tshape0*
_output_shapes
:@
c
Reshape_59/shapeConst*
dtype0*
_output_shapes
:*
valueB:
€€€€€€€€€
o

Reshape_59Reshapev/dense_2/bias/readReshape_59/shape*
_output_shapes
:*
Tshape0*
T0
c
Reshape_60/shapeConst*
valueB:
€€€€€€€€€*
_output_shapes
:*
dtype0
l

Reshape_60Reshapebeta1_power/readReshape_60/shape*
Tshape0*
_output_shapes
:*
T0
c
Reshape_61/shapeConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€
l

Reshape_61Reshapebeta2_power/readReshape_61/shape*
T0*
_output_shapes
:*
Tshape0
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
Reshape_63/shapeConst*
dtype0*
valueB:
€€€€€€€€€*
_output_shapes
:
x

Reshape_63Reshapepi/dense/kernel/Adam_1/readReshape_63/shape*
T0*
Tshape0*
_output_shapes	
:А
c
Reshape_64/shapeConst*
_output_shapes
:*
valueB:
€€€€€€€€€*
dtype0
s

Reshape_64Reshapepi/dense/bias/Adam/readReshape_64/shape*
T0*
Tshape0*
_output_shapes
:@
c
Reshape_65/shapeConst*
dtype0*
_output_shapes
:*
valueB:
€€€€€€€€€
u

Reshape_65Reshapepi/dense/bias/Adam_1/readReshape_65/shape*
Tshape0*
_output_shapes
:@*
T0
c
Reshape_66/shapeConst*
valueB:
€€€€€€€€€*
dtype0*
_output_shapes
:
x

Reshape_66Reshapepi/dense_1/kernel/Adam/readReshape_66/shape*
T0*
_output_shapes	
:А *
Tshape0
c
Reshape_67/shapeConst*
dtype0*
_output_shapes
:*
valueB:
€€€€€€€€€
z

Reshape_67Reshapepi/dense_1/kernel/Adam_1/readReshape_67/shape*
Tshape0*
T0*
_output_shapes	
:А 
c
Reshape_68/shapeConst*
_output_shapes
:*
valueB:
€€€€€€€€€*
dtype0
u

Reshape_68Reshapepi/dense_1/bias/Adam/readReshape_68/shape*
Tshape0*
_output_shapes
:@*
T0
c
Reshape_69/shapeConst*
dtype0*
_output_shapes
:*
valueB:
€€€€€€€€€
w

Reshape_69Reshapepi/dense_1/bias/Adam_1/readReshape_69/shape*
_output_shapes
:@*
T0*
Tshape0
c
Reshape_70/shapeConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€
x

Reshape_70Reshapepi/dense_2/kernel/Adam/readReshape_70/shape*
T0*
Tshape0*
_output_shapes	
:ј
c
Reshape_71/shapeConst*
dtype0*
_output_shapes
:*
valueB:
€€€€€€€€€
z

Reshape_71Reshapepi/dense_2/kernel/Adam_1/readReshape_71/shape*
Tshape0*
T0*
_output_shapes	
:ј
c
Reshape_72/shapeConst*
valueB:
€€€€€€€€€*
dtype0*
_output_shapes
:
u

Reshape_72Reshapepi/dense_2/bias/Adam/readReshape_72/shape*
_output_shapes
:*
T0*
Tshape0
c
Reshape_73/shapeConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€
w

Reshape_73Reshapepi/dense_2/bias/Adam_1/readReshape_73/shape*
T0*
Tshape0*
_output_shapes
:
c
Reshape_74/shapeConst*
_output_shapes
:*
valueB:
€€€€€€€€€*
dtype0
n

Reshape_74Reshapebeta1_power_1/readReshape_74/shape*
Tshape0*
_output_shapes
:*
T0
c
Reshape_75/shapeConst*
dtype0*
valueB:
€€€€€€€€€*
_output_shapes
:
n

Reshape_75Reshapebeta2_power_1/readReshape_75/shape*
Tshape0*
_output_shapes
:*
T0
c
Reshape_76/shapeConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€
u

Reshape_76Reshapev/dense/kernel/Adam/readReshape_76/shape*
T0*
Tshape0*
_output_shapes	
:А
c
Reshape_77/shapeConst*
dtype0*
_output_shapes
:*
valueB:
€€€€€€€€€
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
Tshape0*
_output_shapes
:@*
T0
c
Reshape_79/shapeConst*
_output_shapes
:*
valueB:
€€€€€€€€€*
dtype0
t

Reshape_79Reshapev/dense/bias/Adam_1/readReshape_79/shape*
_output_shapes
:@*
Tshape0*
T0
c
Reshape_80/shapeConst*
valueB:
€€€€€€€€€*
dtype0*
_output_shapes
:
w

Reshape_80Reshapev/dense_1/kernel/Adam/readReshape_80/shape*
Tshape0*
T0*
_output_shapes	
:А 
c
Reshape_81/shapeConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€
y

Reshape_81Reshapev/dense_1/kernel/Adam_1/readReshape_81/shape*
_output_shapes	
:А *
T0*
Tshape0
c
Reshape_82/shapeConst*
valueB:
€€€€€€€€€*
dtype0*
_output_shapes
:
t

Reshape_82Reshapev/dense_1/bias/Adam/readReshape_82/shape*
T0*
_output_shapes
:@*
Tshape0
c
Reshape_83/shapeConst*
dtype0*
_output_shapes
:*
valueB:
€€€€€€€€€
v

Reshape_83Reshapev/dense_1/bias/Adam_1/readReshape_83/shape*
T0*
Tshape0*
_output_shapes
:@
c
Reshape_84/shapeConst*
_output_shapes
:*
valueB:
€€€€€€€€€*
dtype0
v

Reshape_84Reshapev/dense_2/kernel/Adam/readReshape_84/shape*
Tshape0*
T0*
_output_shapes
:@
c
Reshape_85/shapeConst*
dtype0*
valueB:
€€€€€€€€€*
_output_shapes
:
x

Reshape_85Reshapev/dense_2/kernel/Adam_1/readReshape_85/shape*
Tshape0*
T0*
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

Reshape_86Reshapev/dense_2/bias/Adam/readReshape_86/shape*
_output_shapes
:*
T0*
Tshape0
c
Reshape_87/shapeConst*
valueB:
€€€€€€€€€*
dtype0*
_output_shapes
:
v

Reshape_87Reshapev/dense_2/bias/Adam_1/readReshape_87/shape*
Tshape0*
T0*
_output_shapes
:
O
concat_4/axisConst*
value	B : *
dtype0*
_output_shapes
: 
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
Reshape_87concat_4/axis*
N(*

Tidx0*
_output_shapes

:Рё*
T0
h
PyFunc_4PyFuncconcat_4*
Tin
2*
_output_shapes
:*
token
pyfunc_4*
Tout
2
ф
Const_9Const*
_output_shapes
:(*Є
valueЃBЂ("†А  @      @   ј      А  @      @   @            А  А  @   @         @   @   ј   ј               А  А  @   @         @   @   @   @         *
dtype0
S
split_4/split_dimConst*
value	B : *
_output_shapes
: *
dtype0
Ц
split_4SplitVPyFunc_4Const_9split_4/split_dim*

Tlen0*ґ
_output_shapes£
†::::::::::::::::::::::::::::::::::::::::*
	num_split(*
T0
a
Reshape_88/shapeConst*
_output_shapes
:*
valueB"   @   *
dtype0
g

Reshape_88Reshapesplit_4Reshape_88/shape*
T0*
_output_shapes

:@*
Tshape0
Z
Reshape_89/shapeConst*
_output_shapes
:*
dtype0*
valueB:@
e

Reshape_89Reshape	split_4:1Reshape_89/shape*
Tshape0*
T0*
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

:@@*
T0*
Tshape0
Z
Reshape_91/shapeConst*
dtype0*
_output_shapes
:*
valueB:@
e

Reshape_91Reshape	split_4:3Reshape_91/shape*
Tshape0*
T0*
_output_shapes
:@
a
Reshape_92/shapeConst*
dtype0*
valueB"@      *
_output_shapes
:
i

Reshape_92Reshape	split_4:4Reshape_92/shape*
_output_shapes

:@*
Tshape0*
T0
Z
Reshape_93/shapeConst*
valueB:*
_output_shapes
:*
dtype0
e

Reshape_93Reshape	split_4:5Reshape_93/shape*
_output_shapes
:*
T0*
Tshape0
a
Reshape_94/shapeConst*
_output_shapes
:*
dtype0*
valueB"   @   
i

Reshape_94Reshape	split_4:6Reshape_94/shape*
T0*
Tshape0*
_output_shapes

:@
Z
Reshape_95/shapeConst*
valueB:@*
_output_shapes
:*
dtype0
e

Reshape_95Reshape	split_4:7Reshape_95/shape*
Tshape0*
T0*
_output_shapes
:@
a
Reshape_96/shapeConst*
dtype0*
_output_shapes
:*
valueB"@   @   
i

Reshape_96Reshape	split_4:8Reshape_96/shape*
T0*
_output_shapes

:@@*
Tshape0
Z
Reshape_97/shapeConst*
_output_shapes
:*
valueB:@*
dtype0
e

Reshape_97Reshape	split_4:9Reshape_97/shape*
_output_shapes
:@*
Tshape0*
T0
a
Reshape_98/shapeConst*
dtype0*
valueB"@      *
_output_shapes
:
j

Reshape_98Reshape
split_4:10Reshape_98/shape*
Tshape0*
_output_shapes

:@*
T0
Z
Reshape_99/shapeConst*
_output_shapes
:*
dtype0*
valueB:
f

Reshape_99Reshape
split_4:11Reshape_99/shape*
_output_shapes
:*
Tshape0*
T0
T
Reshape_100/shapeConst*
_output_shapes
: *
dtype0*
valueB 
d
Reshape_100Reshape
split_4:12Reshape_100/shape*
_output_shapes
: *
Tshape0*
T0
T
Reshape_101/shapeConst*
_output_shapes
: *
dtype0*
valueB 
d
Reshape_101Reshape
split_4:13Reshape_101/shape*
Tshape0*
T0*
_output_shapes
: 
b
Reshape_102/shapeConst*
dtype0*
_output_shapes
:*
valueB"   @   
l
Reshape_102Reshape
split_4:14Reshape_102/shape*
_output_shapes

:@*
T0*
Tshape0
b
Reshape_103/shapeConst*
valueB"   @   *
dtype0*
_output_shapes
:
l
Reshape_103Reshape
split_4:15Reshape_103/shape*
Tshape0*
_output_shapes

:@*
T0
[
Reshape_104/shapeConst*
valueB:@*
_output_shapes
:*
dtype0
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
:*
dtype0*
valueB:@
h
Reshape_105Reshape
split_4:17Reshape_105/shape*
_output_shapes
:@*
T0*
Tshape0
b
Reshape_106/shapeConst*
valueB"@   @   *
_output_shapes
:*
dtype0
l
Reshape_106Reshape
split_4:18Reshape_106/shape*
T0*
_output_shapes

:@@*
Tshape0
b
Reshape_107/shapeConst*
valueB"@   @   *
dtype0*
_output_shapes
:
l
Reshape_107Reshape
split_4:19Reshape_107/shape*
_output_shapes

:@@*
T0*
Tshape0
[
Reshape_108/shapeConst*
dtype0*
_output_shapes
:*
valueB:@
h
Reshape_108Reshape
split_4:20Reshape_108/shape*
T0*
_output_shapes
:@*
Tshape0
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
valueB"@      *
dtype0
l
Reshape_110Reshape
split_4:22Reshape_110/shape*
T0*
Tshape0*
_output_shapes

:@
b
Reshape_111/shapeConst*
_output_shapes
:*
dtype0*
valueB"@      
l
Reshape_111Reshape
split_4:23Reshape_111/shape*
T0*
Tshape0*
_output_shapes

:@
[
Reshape_112/shapeConst*
dtype0*
valueB:*
_output_shapes
:
h
Reshape_112Reshape
split_4:24Reshape_112/shape*
Tshape0*
_output_shapes
:*
T0
[
Reshape_113/shapeConst*
dtype0*
_output_shapes
:*
valueB:
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
: *
Tshape0*
T0
b
Reshape_116/shapeConst*
_output_shapes
:*
dtype0*
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
Tshape0*
_output_shapes

:@*
T0
[
Reshape_118/shapeConst*
_output_shapes
:*
dtype0*
valueB:@
h
Reshape_118Reshape
split_4:30Reshape_118/shape*
T0*
_output_shapes
:@*
Tshape0
[
Reshape_119/shapeConst*
_output_shapes
:*
valueB:@*
dtype0
h
Reshape_119Reshape
split_4:31Reshape_119/shape*
T0*
Tshape0*
_output_shapes
:@
b
Reshape_120/shapeConst*
dtype0*
_output_shapes
:*
valueB"@   @   
l
Reshape_120Reshape
split_4:32Reshape_120/shape*
Tshape0*
T0*
_output_shapes

:@@
b
Reshape_121/shapeConst*
dtype0*
valueB"@   @   *
_output_shapes
:
l
Reshape_121Reshape
split_4:33Reshape_121/shape*
_output_shapes

:@@*
T0*
Tshape0
[
Reshape_122/shapeConst*
valueB:@*
_output_shapes
:*
dtype0
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
:@*
T0*
Tshape0
b
Reshape_124/shapeConst*
valueB"@      *
dtype0*
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
dtype0*
valueB"@      *
_output_shapes
:
l
Reshape_125Reshape
split_4:37Reshape_125/shape*
Tshape0*
_output_shapes

:@*
T0
[
Reshape_126/shapeConst*
dtype0*
valueB:*
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
Reshape_127/shapeConst*
dtype0*
_output_shapes
:*
valueB:
h
Reshape_127Reshape
split_4:39Reshape_127/shape*
Tshape0*
_output_shapes
:*
T0
¶
	Assign_12Assignpi/dense/kernel
Reshape_88*
_output_shapes

:@*
T0*
validate_shape(*"
_class
loc:@pi/dense/kernel*
use_locking(
Ю
	Assign_13Assignpi/dense/bias
Reshape_89* 
_class
loc:@pi/dense/bias*
T0*
_output_shapes
:@*
use_locking(*
validate_shape(
™
	Assign_14Assignpi/dense_1/kernel
Reshape_90*
T0*
use_locking(*
_output_shapes

:@@*$
_class
loc:@pi/dense_1/kernel*
validate_shape(
Ґ
	Assign_15Assignpi/dense_1/bias
Reshape_91*
validate_shape(*
_output_shapes
:@*"
_class
loc:@pi/dense_1/bias*
T0*
use_locking(
™
	Assign_16Assignpi/dense_2/kernel
Reshape_92*
_output_shapes

:@*
validate_shape(*
use_locking(*
T0*$
_class
loc:@pi/dense_2/kernel
Ґ
	Assign_17Assignpi/dense_2/bias
Reshape_93*"
_class
loc:@pi/dense_2/bias*
T0*
_output_shapes
:*
use_locking(*
validate_shape(
§
	Assign_18Assignv/dense/kernel
Reshape_94*
use_locking(*
_output_shapes

:@*
T0*
validate_shape(*!
_class
loc:@v/dense/kernel
Ь
	Assign_19Assignv/dense/bias
Reshape_95*
_class
loc:@v/dense/bias*
validate_shape(*
use_locking(*
_output_shapes
:@*
T0
®
	Assign_20Assignv/dense_1/kernel
Reshape_96*
T0*
use_locking(*
_output_shapes

:@@*#
_class
loc:@v/dense_1/kernel*
validate_shape(
†
	Assign_21Assignv/dense_1/bias
Reshape_97*
validate_shape(*!
_class
loc:@v/dense_1/bias*
_output_shapes
:@*
T0*
use_locking(
®
	Assign_22Assignv/dense_2/kernel
Reshape_98*
_output_shapes

:@*
T0*
use_locking(*
validate_shape(*#
_class
loc:@v/dense_2/kernel
†
	Assign_23Assignv/dense_2/bias
Reshape_99*
T0*
use_locking(*
_output_shapes
:*!
_class
loc:@v/dense_2/bias*
validate_shape(
Щ
	Assign_24Assignbeta1_powerReshape_100* 
_class
loc:@pi/dense/bias*
T0*
use_locking(*
_output_shapes
: *
validate_shape(
Щ
	Assign_25Assignbeta2_powerReshape_101*
_output_shapes
: *
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
use_locking(
ђ
	Assign_26Assignpi/dense/kernel/AdamReshape_102*
_output_shapes

:@*
validate_shape(*
T0*
use_locking(*"
_class
loc:@pi/dense/kernel
Ѓ
	Assign_27Assignpi/dense/kernel/Adam_1Reshape_103*"
_class
loc:@pi/dense/kernel*
use_locking(*
validate_shape(*
T0*
_output_shapes

:@
§
	Assign_28Assignpi/dense/bias/AdamReshape_104*
T0*
use_locking(* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
:@
¶
	Assign_29Assignpi/dense/bias/Adam_1Reshape_105*
_output_shapes
:@*
validate_shape(*
T0*
use_locking(* 
_class
loc:@pi/dense/bias
∞
	Assign_30Assignpi/dense_1/kernel/AdamReshape_106*
validate_shape(*
use_locking(*$
_class
loc:@pi/dense_1/kernel*
T0*
_output_shapes

:@@
≤
	Assign_31Assignpi/dense_1/kernel/Adam_1Reshape_107*$
_class
loc:@pi/dense_1/kernel*
T0*
validate_shape(*
_output_shapes

:@@*
use_locking(
®
	Assign_32Assignpi/dense_1/bias/AdamReshape_108*
use_locking(*
validate_shape(*
T0*"
_class
loc:@pi/dense_1/bias*
_output_shapes
:@
™
	Assign_33Assignpi/dense_1/bias/Adam_1Reshape_109*
validate_shape(*
use_locking(*
_output_shapes
:@*"
_class
loc:@pi/dense_1/bias*
T0
∞
	Assign_34Assignpi/dense_2/kernel/AdamReshape_110*
use_locking(*$
_class
loc:@pi/dense_2/kernel*
_output_shapes

:@*
T0*
validate_shape(
≤
	Assign_35Assignpi/dense_2/kernel/Adam_1Reshape_111*
_output_shapes

:@*
use_locking(*$
_class
loc:@pi/dense_2/kernel*
T0*
validate_shape(
®
	Assign_36Assignpi/dense_2/bias/AdamReshape_112*
T0*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(
™
	Assign_37Assignpi/dense_2/bias/Adam_1Reshape_113*
_output_shapes
:*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
use_locking(*
T0
Ъ
	Assign_38Assignbeta1_power_1Reshape_114*
_class
loc:@v/dense/bias*
validate_shape(*
T0*
use_locking(*
_output_shapes
: 
Ъ
	Assign_39Assignbeta2_power_1Reshape_115*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
: *
T0*
use_locking(
™
	Assign_40Assignv/dense/kernel/AdamReshape_116*
T0*
_output_shapes

:@*!
_class
loc:@v/dense/kernel*
validate_shape(*
use_locking(
ђ
	Assign_41Assignv/dense/kernel/Adam_1Reshape_117*
use_locking(*!
_class
loc:@v/dense/kernel*
T0*
validate_shape(*
_output_shapes

:@
Ґ
	Assign_42Assignv/dense/bias/AdamReshape_118*
_output_shapes
:@*
use_locking(*
validate_shape(*
_class
loc:@v/dense/bias*
T0
§
	Assign_43Assignv/dense/bias/Adam_1Reshape_119*
_class
loc:@v/dense/bias*
T0*
_output_shapes
:@*
validate_shape(*
use_locking(
Ѓ
	Assign_44Assignv/dense_1/kernel/AdamReshape_120*
_output_shapes

:@@*
T0*#
_class
loc:@v/dense_1/kernel*
use_locking(*
validate_shape(
∞
	Assign_45Assignv/dense_1/kernel/Adam_1Reshape_121*
validate_shape(*
T0*
_output_shapes

:@@*
use_locking(*#
_class
loc:@v/dense_1/kernel
¶
	Assign_46Assignv/dense_1/bias/AdamReshape_122*
_output_shapes
:@*
T0*
validate_shape(*
use_locking(*!
_class
loc:@v/dense_1/bias
®
	Assign_47Assignv/dense_1/bias/Adam_1Reshape_123*
T0*
_output_shapes
:@*!
_class
loc:@v/dense_1/bias*
use_locking(*
validate_shape(
Ѓ
	Assign_48Assignv/dense_2/kernel/AdamReshape_124*
_output_shapes

:@*
use_locking(*
T0*#
_class
loc:@v/dense_2/kernel*
validate_shape(
∞
	Assign_49Assignv/dense_2/kernel/Adam_1Reshape_125*
validate_shape(*#
_class
loc:@v/dense_2/kernel*
T0*
use_locking(*
_output_shapes

:@
¶
	Assign_50Assignv/dense_2/bias/AdamReshape_126*
use_locking(*
T0*
_output_shapes
:*!
_class
loc:@v/dense_2/bias*
validate_shape(
®
	Assign_51Assignv/dense_2/bias/Adam_1Reshape_127*
_output_shapes
:*
use_locking(*!
_class
loc:@v/dense_2/bias*
validate_shape(*
T0
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
dtype0*
_output_shapes
: *
shape: 
e

save/ConstPlaceholderWithDefaultsave/filename*
dtype0*
shape: *
_output_shapes
: 
Д
save/StringJoin/inputs_1Const*<
value3B1 B+_temp_bff35ad25c5f430fa41597af754d2874/part*
dtype0*
_output_shapes
: 
u
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
Q
save/num_shardsConst*
_output_shapes
: *
dtype0*
value	B :
\
save/ShardedFilename/shardConst*
value	B : *
_output_shapes
: *
dtype0
}
save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards*
_output_shapes
: 
И
save/SaveV2/tensor_namesConst*
_output_shapes
:(*
dtype0*ї
value±BЃ(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1
≥
save/SaveV2/shape_and_slicesConst*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:(
Ѕ
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1v/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1v/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1v/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1v/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1v/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1v/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1*6
dtypes,
*2(
С
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2*
_output_shapes
: *'
_class
loc:@save/ShardedFilename*
T0
Э
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency*
_output_shapes
:*

axis *
N*
T0
}
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const*
delete_old_dirs(
z
save/IdentityIdentity
save/Const^save/MergeV2Checkpoints^save/control_dependency*
T0*
_output_shapes
: 
Л
save/RestoreV2/tensor_namesConst*
dtype0*ї
value±BЃ(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
_output_shapes
:(
ґ
save/RestoreV2/shape_and_slicesConst*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:(*
dtype0
÷
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*ґ
_output_shapes£
†::::::::::::::::::::::::::::::::::::::::*6
dtypes,
*2(
Ю
save/AssignAssignbeta1_powersave/RestoreV2*
T0*
_output_shapes
: *
use_locking(*
validate_shape(* 
_class
loc:@pi/dense/bias
£
save/Assign_1Assignbeta1_power_1save/RestoreV2:1*
T0*
_class
loc:@v/dense/bias*
use_locking(*
_output_shapes
: *
validate_shape(
Ґ
save/Assign_2Assignbeta2_powersave/RestoreV2:2*
_output_shapes
: *
validate_shape(* 
_class
loc:@pi/dense/bias*
T0*
use_locking(
£
save/Assign_3Assignbeta2_power_1save/RestoreV2:3*
_class
loc:@v/dense/bias*
use_locking(*
T0*
_output_shapes
: *
validate_shape(
®
save/Assign_4Assignpi/dense/biassave/RestoreV2:4* 
_class
loc:@pi/dense/bias*
use_locking(*
validate_shape(*
T0*
_output_shapes
:@
≠
save/Assign_5Assignpi/dense/bias/Adamsave/RestoreV2:5*
validate_shape(*
use_locking(* 
_class
loc:@pi/dense/bias*
_output_shapes
:@*
T0
ѓ
save/Assign_6Assignpi/dense/bias/Adam_1save/RestoreV2:6*
T0* 
_class
loc:@pi/dense/bias*
_output_shapes
:@*
use_locking(*
validate_shape(
∞
save/Assign_7Assignpi/dense/kernelsave/RestoreV2:7*
_output_shapes

:@*
use_locking(*
T0*
validate_shape(*"
_class
loc:@pi/dense/kernel
µ
save/Assign_8Assignpi/dense/kernel/Adamsave/RestoreV2:8*"
_class
loc:@pi/dense/kernel*
_output_shapes

:@*
use_locking(*
validate_shape(*
T0
Ј
save/Assign_9Assignpi/dense/kernel/Adam_1save/RestoreV2:9*
use_locking(*
_output_shapes

:@*
T0*
validate_shape(*"
_class
loc:@pi/dense/kernel
Ѓ
save/Assign_10Assignpi/dense_1/biassave/RestoreV2:10*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*"
_class
loc:@pi/dense_1/bias
≥
save/Assign_11Assignpi/dense_1/bias/Adamsave/RestoreV2:11*"
_class
loc:@pi/dense_1/bias*
_output_shapes
:@*
validate_shape(*
use_locking(*
T0
µ
save/Assign_12Assignpi/dense_1/bias/Adam_1save/RestoreV2:12*
_output_shapes
:@*
use_locking(*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
T0
ґ
save/Assign_13Assignpi/dense_1/kernelsave/RestoreV2:13*
_output_shapes

:@@*
validate_shape(*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel
ї
save/Assign_14Assignpi/dense_1/kernel/Adamsave/RestoreV2:14*
_output_shapes

:@@*
validate_shape(*
T0*$
_class
loc:@pi/dense_1/kernel*
use_locking(
љ
save/Assign_15Assignpi/dense_1/kernel/Adam_1save/RestoreV2:15*
use_locking(*
T0*
validate_shape(*$
_class
loc:@pi/dense_1/kernel*
_output_shapes

:@@
Ѓ
save/Assign_16Assignpi/dense_2/biassave/RestoreV2:16*
validate_shape(*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:
≥
save/Assign_17Assignpi/dense_2/bias/Adamsave/RestoreV2:17*
validate_shape(*
T0*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:*
use_locking(
µ
save/Assign_18Assignpi/dense_2/bias/Adam_1save/RestoreV2:18*
validate_shape(*
_output_shapes
:*"
_class
loc:@pi/dense_2/bias*
use_locking(*
T0
ґ
save/Assign_19Assignpi/dense_2/kernelsave/RestoreV2:19*
use_locking(*$
_class
loc:@pi/dense_2/kernel*
_output_shapes

:@*
validate_shape(*
T0
ї
save/Assign_20Assignpi/dense_2/kernel/Adamsave/RestoreV2:20*
validate_shape(*
use_locking(*
T0*$
_class
loc:@pi/dense_2/kernel*
_output_shapes

:@
љ
save/Assign_21Assignpi/dense_2/kernel/Adam_1save/RestoreV2:21*
T0*
use_locking(*
_output_shapes

:@*$
_class
loc:@pi/dense_2/kernel*
validate_shape(
®
save/Assign_22Assignv/dense/biassave/RestoreV2:22*
T0*
_output_shapes
:@*
use_locking(*
validate_shape(*
_class
loc:@v/dense/bias
≠
save/Assign_23Assignv/dense/bias/Adamsave/RestoreV2:23*
validate_shape(*
_class
loc:@v/dense/bias*
T0*
_output_shapes
:@*
use_locking(
ѓ
save/Assign_24Assignv/dense/bias/Adam_1save/RestoreV2:24*
_class
loc:@v/dense/bias*
_output_shapes
:@*
T0*
validate_shape(*
use_locking(
∞
save/Assign_25Assignv/dense/kernelsave/RestoreV2:25*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0*!
_class
loc:@v/dense/kernel
µ
save/Assign_26Assignv/dense/kernel/Adamsave/RestoreV2:26*!
_class
loc:@v/dense/kernel*
_output_shapes

:@*
T0*
use_locking(*
validate_shape(
Ј
save/Assign_27Assignv/dense/kernel/Adam_1save/RestoreV2:27*
validate_shape(*
use_locking(*!
_class
loc:@v/dense/kernel*
T0*
_output_shapes

:@
ђ
save/Assign_28Assignv/dense_1/biassave/RestoreV2:28*
validate_shape(*
T0*
use_locking(*!
_class
loc:@v/dense_1/bias*
_output_shapes
:@
±
save/Assign_29Assignv/dense_1/bias/Adamsave/RestoreV2:29*
T0*
validate_shape(*
use_locking(*!
_class
loc:@v/dense_1/bias*
_output_shapes
:@
≥
save/Assign_30Assignv/dense_1/bias/Adam_1save/RestoreV2:30*
_output_shapes
:@*
use_locking(*
validate_shape(*
T0*!
_class
loc:@v/dense_1/bias
і
save/Assign_31Assignv/dense_1/kernelsave/RestoreV2:31*
validate_shape(*
use_locking(*#
_class
loc:@v/dense_1/kernel*
T0*
_output_shapes

:@@
є
save/Assign_32Assignv/dense_1/kernel/Adamsave/RestoreV2:32*
_output_shapes

:@@*#
_class
loc:@v/dense_1/kernel*
validate_shape(*
use_locking(*
T0
ї
save/Assign_33Assignv/dense_1/kernel/Adam_1save/RestoreV2:33*
T0*
validate_shape(*
use_locking(*#
_class
loc:@v/dense_1/kernel*
_output_shapes

:@@
ђ
save/Assign_34Assignv/dense_2/biassave/RestoreV2:34*
_output_shapes
:*
T0*
use_locking(*!
_class
loc:@v/dense_2/bias*
validate_shape(
±
save/Assign_35Assignv/dense_2/bias/Adamsave/RestoreV2:35*
_output_shapes
:*
validate_shape(*
T0*!
_class
loc:@v/dense_2/bias*
use_locking(
≥
save/Assign_36Assignv/dense_2/bias/Adam_1save/RestoreV2:36*
T0*
_output_shapes
:*
validate_shape(*!
_class
loc:@v/dense_2/bias*
use_locking(
і
save/Assign_37Assignv/dense_2/kernelsave/RestoreV2:37*
validate_shape(*
_output_shapes

:@*
T0*
use_locking(*#
_class
loc:@v/dense_2/kernel
є
save/Assign_38Assignv/dense_2/kernel/Adamsave/RestoreV2:38*
use_locking(*
T0*
_output_shapes

:@*
validate_shape(*#
_class
loc:@v/dense_2/kernel
ї
save/Assign_39Assignv/dense_2/kernel/Adam_1save/RestoreV2:39*#
_class
loc:@v/dense_2/kernel*
validate_shape(*
use_locking(*
T0*
_output_shapes

:@
ґ
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
save_1/filenamePlaceholderWithDefaultsave_1/filename/input*
_output_shapes
: *
dtype0*
shape: 
i
save_1/ConstPlaceholderWithDefaultsave_1/filename*
shape: *
dtype0*
_output_shapes
: 
Ж
save_1/StringJoin/inputs_1Const*<
value3B1 B+_temp_a5b805c678424e1784c79438f3ddf057/part*
_output_shapes
: *
dtype0
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
save_1/ShardedFilename/shardConst*
value	B : *
dtype0*
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
save_1/SaveV2/shape_and_slicesConst*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:(*
dtype0
…
save_1/SaveV2SaveV2save_1/ShardedFilenamesave_1/SaveV2/tensor_namessave_1/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1v/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1v/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1v/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1v/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1v/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1v/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1*6
dtypes,
*2(
Щ
save_1/control_dependencyIdentitysave_1/ShardedFilename^save_1/SaveV2*)
_class
loc:@save_1/ShardedFilename*
T0*
_output_shapes
: 
£
-save_1/MergeV2Checkpoints/checkpoint_prefixesPacksave_1/ShardedFilename^save_1/control_dependency*
T0*
_output_shapes
:*

axis *
N
Г
save_1/MergeV2CheckpointsMergeV2Checkpoints-save_1/MergeV2Checkpoints/checkpoint_prefixessave_1/Const*
delete_old_dirs(
В
save_1/IdentityIdentitysave_1/Const^save_1/MergeV2Checkpoints^save_1/control_dependency*
T0*
_output_shapes
: 
Н
save_1/RestoreV2/tensor_namesConst*
_output_shapes
:(*
dtype0*ї
value±BЃ(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1
Є
!save_1/RestoreV2/shape_and_slicesConst*
_output_shapes
:(*
dtype0*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
ё
save_1/RestoreV2	RestoreV2save_1/Constsave_1/RestoreV2/tensor_names!save_1/RestoreV2/shape_and_slices*6
dtypes,
*2(*ґ
_output_shapes£
†::::::::::::::::::::::::::::::::::::::::
Ґ
save_1/AssignAssignbeta1_powersave_1/RestoreV2*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
_output_shapes
: *
validate_shape(
І
save_1/Assign_1Assignbeta1_power_1save_1/RestoreV2:1*
T0*
validate_shape(*
_class
loc:@v/dense/bias*
use_locking(*
_output_shapes
: 
¶
save_1/Assign_2Assignbeta2_powersave_1/RestoreV2:2*
validate_shape(* 
_class
loc:@pi/dense/bias*
T0*
_output_shapes
: *
use_locking(
І
save_1/Assign_3Assignbeta2_power_1save_1/RestoreV2:3*
T0*
validate_shape(*
_class
loc:@v/dense/bias*
_output_shapes
: *
use_locking(
ђ
save_1/Assign_4Assignpi/dense/biassave_1/RestoreV2:4*
_output_shapes
:@*
T0*
validate_shape(*
use_locking(* 
_class
loc:@pi/dense/bias
±
save_1/Assign_5Assignpi/dense/bias/Adamsave_1/RestoreV2:5* 
_class
loc:@pi/dense/bias*
use_locking(*
T0*
validate_shape(*
_output_shapes
:@
≥
save_1/Assign_6Assignpi/dense/bias/Adam_1save_1/RestoreV2:6*
use_locking(*
_output_shapes
:@*
validate_shape(* 
_class
loc:@pi/dense/bias*
T0
і
save_1/Assign_7Assignpi/dense/kernelsave_1/RestoreV2:7*
T0*
validate_shape(*
use_locking(*
_output_shapes

:@*"
_class
loc:@pi/dense/kernel
є
save_1/Assign_8Assignpi/dense/kernel/Adamsave_1/RestoreV2:8*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes

:@*
use_locking(
ї
save_1/Assign_9Assignpi/dense/kernel/Adam_1save_1/RestoreV2:9*
use_locking(*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes

:@*
T0
≤
save_1/Assign_10Assignpi/dense_1/biassave_1/RestoreV2:10*
_output_shapes
:@*
use_locking(*"
_class
loc:@pi/dense_1/bias*
T0*
validate_shape(
Ј
save_1/Assign_11Assignpi/dense_1/bias/Adamsave_1/RestoreV2:11*
validate_shape(*"
_class
loc:@pi/dense_1/bias*
use_locking(*
T0*
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
save_1/Assign_13Assignpi/dense_1/kernelsave_1/RestoreV2:13*
_output_shapes

:@@*
validate_shape(*$
_class
loc:@pi/dense_1/kernel*
use_locking(*
T0
њ
save_1/Assign_14Assignpi/dense_1/kernel/Adamsave_1/RestoreV2:14*
validate_shape(*
T0*
use_locking(*
_output_shapes

:@@*$
_class
loc:@pi/dense_1/kernel
Ѕ
save_1/Assign_15Assignpi/dense_1/kernel/Adam_1save_1/RestoreV2:15*
T0*$
_class
loc:@pi/dense_1/kernel*
validate_shape(*
use_locking(*
_output_shapes

:@@
≤
save_1/Assign_16Assignpi/dense_2/biassave_1/RestoreV2:16*
_output_shapes
:*
use_locking(*"
_class
loc:@pi/dense_2/bias*
T0*
validate_shape(
Ј
save_1/Assign_17Assignpi/dense_2/bias/Adamsave_1/RestoreV2:17*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:*
validate_shape(
є
save_1/Assign_18Assignpi/dense_2/bias/Adam_1save_1/RestoreV2:18*"
_class
loc:@pi/dense_2/bias*
T0*
validate_shape(*
use_locking(*
_output_shapes
:
Ї
save_1/Assign_19Assignpi/dense_2/kernelsave_1/RestoreV2:19*
use_locking(*
validate_shape(*
_output_shapes

:@*$
_class
loc:@pi/dense_2/kernel*
T0
њ
save_1/Assign_20Assignpi/dense_2/kernel/Adamsave_1/RestoreV2:20*
_output_shapes

:@*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
use_locking(*
T0
Ѕ
save_1/Assign_21Assignpi/dense_2/kernel/Adam_1save_1/RestoreV2:21*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0*$
_class
loc:@pi/dense_2/kernel
ђ
save_1/Assign_22Assignv/dense/biassave_1/RestoreV2:22*
T0*
_class
loc:@v/dense/bias*
_output_shapes
:@*
use_locking(*
validate_shape(
±
save_1/Assign_23Assignv/dense/bias/Adamsave_1/RestoreV2:23*
_class
loc:@v/dense/bias*
validate_shape(*
use_locking(*
_output_shapes
:@*
T0
≥
save_1/Assign_24Assignv/dense/bias/Adam_1save_1/RestoreV2:24*
use_locking(*
_class
loc:@v/dense/bias*
validate_shape(*
T0*
_output_shapes
:@
і
save_1/Assign_25Assignv/dense/kernelsave_1/RestoreV2:25*
validate_shape(*
T0*!
_class
loc:@v/dense/kernel*
_output_shapes

:@*
use_locking(
є
save_1/Assign_26Assignv/dense/kernel/Adamsave_1/RestoreV2:26*
validate_shape(*!
_class
loc:@v/dense/kernel*
T0*
_output_shapes

:@*
use_locking(
ї
save_1/Assign_27Assignv/dense/kernel/Adam_1save_1/RestoreV2:27*
_output_shapes

:@*
validate_shape(*!
_class
loc:@v/dense/kernel*
use_locking(*
T0
∞
save_1/Assign_28Assignv/dense_1/biassave_1/RestoreV2:28*!
_class
loc:@v/dense_1/bias*
use_locking(*
validate_shape(*
_output_shapes
:@*
T0
µ
save_1/Assign_29Assignv/dense_1/bias/Adamsave_1/RestoreV2:29*
_output_shapes
:@*!
_class
loc:@v/dense_1/bias*
use_locking(*
T0*
validate_shape(
Ј
save_1/Assign_30Assignv/dense_1/bias/Adam_1save_1/RestoreV2:30*
use_locking(*!
_class
loc:@v/dense_1/bias*
validate_shape(*
T0*
_output_shapes
:@
Є
save_1/Assign_31Assignv/dense_1/kernelsave_1/RestoreV2:31*
use_locking(*
_output_shapes

:@@*#
_class
loc:@v/dense_1/kernel*
validate_shape(*
T0
љ
save_1/Assign_32Assignv/dense_1/kernel/Adamsave_1/RestoreV2:32*
T0*
use_locking(*
validate_shape(*
_output_shapes

:@@*#
_class
loc:@v/dense_1/kernel
њ
save_1/Assign_33Assignv/dense_1/kernel/Adam_1save_1/RestoreV2:33*
validate_shape(*#
_class
loc:@v/dense_1/kernel*
T0*
use_locking(*
_output_shapes

:@@
∞
save_1/Assign_34Assignv/dense_2/biassave_1/RestoreV2:34*
T0*
use_locking(*!
_class
loc:@v/dense_2/bias*
validate_shape(*
_output_shapes
:
µ
save_1/Assign_35Assignv/dense_2/bias/Adamsave_1/RestoreV2:35*
T0*
validate_shape(*!
_class
loc:@v/dense_2/bias*
use_locking(*
_output_shapes
:
Ј
save_1/Assign_36Assignv/dense_2/bias/Adam_1save_1/RestoreV2:36*
T0*!
_class
loc:@v/dense_2/bias*
use_locking(*
validate_shape(*
_output_shapes
:
Є
save_1/Assign_37Assignv/dense_2/kernelsave_1/RestoreV2:37*#
_class
loc:@v/dense_2/kernel*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0
љ
save_1/Assign_38Assignv/dense_2/kernel/Adamsave_1/RestoreV2:38*#
_class
loc:@v/dense_2/kernel*
use_locking(*
T0*
_output_shapes

:@*
validate_shape(
њ
save_1/Assign_39Assignv/dense_2/kernel/Adam_1save_1/RestoreV2:39*
validate_shape(*
use_locking(*#
_class
loc:@v/dense_2/kernel*
_output_shapes

:@*
T0
И
save_1/restore_shardNoOp^save_1/Assign^save_1/Assign_1^save_1/Assign_10^save_1/Assign_11^save_1/Assign_12^save_1/Assign_13^save_1/Assign_14^save_1/Assign_15^save_1/Assign_16^save_1/Assign_17^save_1/Assign_18^save_1/Assign_19^save_1/Assign_2^save_1/Assign_20^save_1/Assign_21^save_1/Assign_22^save_1/Assign_23^save_1/Assign_24^save_1/Assign_25^save_1/Assign_26^save_1/Assign_27^save_1/Assign_28^save_1/Assign_29^save_1/Assign_3^save_1/Assign_30^save_1/Assign_31^save_1/Assign_32^save_1/Assign_33^save_1/Assign_34^save_1/Assign_35^save_1/Assign_36^save_1/Assign_37^save_1/Assign_38^save_1/Assign_39^save_1/Assign_4^save_1/Assign_5^save_1/Assign_6^save_1/Assign_7^save_1/Assign_8^save_1/Assign_9
1
save_1/restore_allNoOp^save_1/restore_shard
[
save_2/filename/inputConst*
valueB Bmodel*
_output_shapes
: *
dtype0
r
save_2/filenamePlaceholderWithDefaultsave_2/filename/input*
dtype0*
_output_shapes
: *
shape: 
i
save_2/ConstPlaceholderWithDefaultsave_2/filename*
shape: *
dtype0*
_output_shapes
: 
Ж
save_2/StringJoin/inputs_1Const*<
value3B1 B+_temp_93bf199e2b62456f8fec395bfb3d0107/part*
dtype0*
_output_shapes
: 
{
save_2/StringJoin
StringJoinsave_2/Constsave_2/StringJoin/inputs_1*
	separator *
_output_shapes
: *
N
S
save_2/num_shardsConst*
_output_shapes
: *
dtype0*
value	B :
^
save_2/ShardedFilename/shardConst*
dtype0*
_output_shapes
: *
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
:(*
dtype0*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
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
-save_2/MergeV2Checkpoints/checkpoint_prefixesPacksave_2/ShardedFilename^save_2/control_dependency*
T0*

axis *
_output_shapes
:*
N
Г
save_2/MergeV2CheckpointsMergeV2Checkpoints-save_2/MergeV2Checkpoints/checkpoint_prefixessave_2/Const*
delete_old_dirs(
В
save_2/IdentityIdentitysave_2/Const^save_2/MergeV2Checkpoints^save_2/control_dependency*
T0*
_output_shapes
: 
Н
save_2/RestoreV2/tensor_namesConst*ї
value±BЃ(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:(
Є
!save_2/RestoreV2/shape_and_slicesConst*
dtype0*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:(
ё
save_2/RestoreV2	RestoreV2save_2/Constsave_2/RestoreV2/tensor_names!save_2/RestoreV2/shape_and_slices*6
dtypes,
*2(*ґ
_output_shapes£
†::::::::::::::::::::::::::::::::::::::::
Ґ
save_2/AssignAssignbeta1_powersave_2/RestoreV2* 
_class
loc:@pi/dense/bias*
use_locking(*
T0*
_output_shapes
: *
validate_shape(
І
save_2/Assign_1Assignbeta1_power_1save_2/RestoreV2:1*
use_locking(*
validate_shape(*
_class
loc:@v/dense/bias*
_output_shapes
: *
T0
¶
save_2/Assign_2Assignbeta2_powersave_2/RestoreV2:2*
use_locking(* 
_class
loc:@pi/dense/bias*
validate_shape(*
T0*
_output_shapes
: 
І
save_2/Assign_3Assignbeta2_power_1save_2/RestoreV2:3*
_class
loc:@v/dense/bias*
_output_shapes
: *
T0*
validate_shape(*
use_locking(
ђ
save_2/Assign_4Assignpi/dense/biassave_2/RestoreV2:4*
use_locking(* 
_class
loc:@pi/dense/bias*
validate_shape(*
T0*
_output_shapes
:@
±
save_2/Assign_5Assignpi/dense/bias/Adamsave_2/RestoreV2:5*
use_locking(*
_output_shapes
:@* 
_class
loc:@pi/dense/bias*
T0*
validate_shape(
≥
save_2/Assign_6Assignpi/dense/bias/Adam_1save_2/RestoreV2:6* 
_class
loc:@pi/dense/bias*
validate_shape(*
T0*
_output_shapes
:@*
use_locking(
і
save_2/Assign_7Assignpi/dense/kernelsave_2/RestoreV2:7*
_output_shapes

:@*
T0*"
_class
loc:@pi/dense/kernel*
use_locking(*
validate_shape(
є
save_2/Assign_8Assignpi/dense/kernel/Adamsave_2/RestoreV2:8*
T0*
use_locking(*
_output_shapes

:@*"
_class
loc:@pi/dense/kernel*
validate_shape(
ї
save_2/Assign_9Assignpi/dense/kernel/Adam_1save_2/RestoreV2:9*"
_class
loc:@pi/dense/kernel*
_output_shapes

:@*
use_locking(*
validate_shape(*
T0
≤
save_2/Assign_10Assignpi/dense_1/biassave_2/RestoreV2:10*
validate_shape(*"
_class
loc:@pi/dense_1/bias*
T0*
_output_shapes
:@*
use_locking(
Ј
save_2/Assign_11Assignpi/dense_1/bias/Adamsave_2/RestoreV2:11*
use_locking(*"
_class
loc:@pi/dense_1/bias*
_output_shapes
:@*
validate_shape(*
T0
є
save_2/Assign_12Assignpi/dense_1/bias/Adam_1save_2/RestoreV2:12*
use_locking(*
T0*
validate_shape(*"
_class
loc:@pi/dense_1/bias*
_output_shapes
:@
Ї
save_2/Assign_13Assignpi/dense_1/kernelsave_2/RestoreV2:13*
use_locking(*
T0*
_output_shapes

:@@*
validate_shape(*$
_class
loc:@pi/dense_1/kernel
њ
save_2/Assign_14Assignpi/dense_1/kernel/Adamsave_2/RestoreV2:14*$
_class
loc:@pi/dense_1/kernel*
use_locking(*
_output_shapes

:@@*
T0*
validate_shape(
Ѕ
save_2/Assign_15Assignpi/dense_1/kernel/Adam_1save_2/RestoreV2:15*
_output_shapes

:@@*$
_class
loc:@pi/dense_1/kernel*
validate_shape(*
use_locking(*
T0
≤
save_2/Assign_16Assignpi/dense_2/biassave_2/RestoreV2:16*
T0*
validate_shape(*
_output_shapes
:*"
_class
loc:@pi/dense_2/bias*
use_locking(
Ј
save_2/Assign_17Assignpi/dense_2/bias/Adamsave_2/RestoreV2:17*
validate_shape(*
T0*"
_class
loc:@pi/dense_2/bias*
use_locking(*
_output_shapes
:
є
save_2/Assign_18Assignpi/dense_2/bias/Adam_1save_2/RestoreV2:18*
T0*
use_locking(*
_output_shapes
:*"
_class
loc:@pi/dense_2/bias*
validate_shape(
Ї
save_2/Assign_19Assignpi/dense_2/kernelsave_2/RestoreV2:19*
_output_shapes

:@*
use_locking(*
T0*
validate_shape(*$
_class
loc:@pi/dense_2/kernel
њ
save_2/Assign_20Assignpi/dense_2/kernel/Adamsave_2/RestoreV2:20*
_output_shapes

:@*
T0*
use_locking(*$
_class
loc:@pi/dense_2/kernel*
validate_shape(
Ѕ
save_2/Assign_21Assignpi/dense_2/kernel/Adam_1save_2/RestoreV2:21*
T0*
_output_shapes

:@*
use_locking(*
validate_shape(*$
_class
loc:@pi/dense_2/kernel
ђ
save_2/Assign_22Assignv/dense/biassave_2/RestoreV2:22*
validate_shape(*
use_locking(*
_output_shapes
:@*
T0*
_class
loc:@v/dense/bias
±
save_2/Assign_23Assignv/dense/bias/Adamsave_2/RestoreV2:23*
T0*
use_locking(*
_class
loc:@v/dense/bias*
_output_shapes
:@*
validate_shape(
≥
save_2/Assign_24Assignv/dense/bias/Adam_1save_2/RestoreV2:24*
_output_shapes
:@*
_class
loc:@v/dense/bias*
T0*
use_locking(*
validate_shape(
і
save_2/Assign_25Assignv/dense/kernelsave_2/RestoreV2:25*
use_locking(*
_output_shapes

:@*!
_class
loc:@v/dense/kernel*
validate_shape(*
T0
є
save_2/Assign_26Assignv/dense/kernel/Adamsave_2/RestoreV2:26*
_output_shapes

:@*!
_class
loc:@v/dense/kernel*
use_locking(*
validate_shape(*
T0
ї
save_2/Assign_27Assignv/dense/kernel/Adam_1save_2/RestoreV2:27*
validate_shape(*
_output_shapes

:@*!
_class
loc:@v/dense/kernel*
use_locking(*
T0
∞
save_2/Assign_28Assignv/dense_1/biassave_2/RestoreV2:28*!
_class
loc:@v/dense_1/bias*
validate_shape(*
T0*
use_locking(*
_output_shapes
:@
µ
save_2/Assign_29Assignv/dense_1/bias/Adamsave_2/RestoreV2:29*
use_locking(*
T0*
_output_shapes
:@*
validate_shape(*!
_class
loc:@v/dense_1/bias
Ј
save_2/Assign_30Assignv/dense_1/bias/Adam_1save_2/RestoreV2:30*
T0*
_output_shapes
:@*
validate_shape(*!
_class
loc:@v/dense_1/bias*
use_locking(
Є
save_2/Assign_31Assignv/dense_1/kernelsave_2/RestoreV2:31*
T0*
_output_shapes

:@@*
validate_shape(*#
_class
loc:@v/dense_1/kernel*
use_locking(
љ
save_2/Assign_32Assignv/dense_1/kernel/Adamsave_2/RestoreV2:32*
use_locking(*
T0*
_output_shapes

:@@*#
_class
loc:@v/dense_1/kernel*
validate_shape(
њ
save_2/Assign_33Assignv/dense_1/kernel/Adam_1save_2/RestoreV2:33*#
_class
loc:@v/dense_1/kernel*
validate_shape(*
_output_shapes

:@@*
T0*
use_locking(
∞
save_2/Assign_34Assignv/dense_2/biassave_2/RestoreV2:34*
use_locking(*!
_class
loc:@v/dense_2/bias*
_output_shapes
:*
validate_shape(*
T0
µ
save_2/Assign_35Assignv/dense_2/bias/Adamsave_2/RestoreV2:35*!
_class
loc:@v/dense_2/bias*
validate_shape(*
_output_shapes
:*
T0*
use_locking(
Ј
save_2/Assign_36Assignv/dense_2/bias/Adam_1save_2/RestoreV2:36*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*!
_class
loc:@v/dense_2/bias
Є
save_2/Assign_37Assignv/dense_2/kernelsave_2/RestoreV2:37*
T0*
validate_shape(*
use_locking(*
_output_shapes

:@*#
_class
loc:@v/dense_2/kernel
љ
save_2/Assign_38Assignv/dense_2/kernel/Adamsave_2/RestoreV2:38*#
_class
loc:@v/dense_2/kernel*
T0*
_output_shapes

:@*
use_locking(*
validate_shape(
њ
save_2/Assign_39Assignv/dense_2/kernel/Adam_1save_2/RestoreV2:39*
T0*#
_class
loc:@v/dense_2/kernel*
use_locking(*
validate_shape(*
_output_shapes

:@
И
save_2/restore_shardNoOp^save_2/Assign^save_2/Assign_1^save_2/Assign_10^save_2/Assign_11^save_2/Assign_12^save_2/Assign_13^save_2/Assign_14^save_2/Assign_15^save_2/Assign_16^save_2/Assign_17^save_2/Assign_18^save_2/Assign_19^save_2/Assign_2^save_2/Assign_20^save_2/Assign_21^save_2/Assign_22^save_2/Assign_23^save_2/Assign_24^save_2/Assign_25^save_2/Assign_26^save_2/Assign_27^save_2/Assign_28^save_2/Assign_29^save_2/Assign_3^save_2/Assign_30^save_2/Assign_31^save_2/Assign_32^save_2/Assign_33^save_2/Assign_34^save_2/Assign_35^save_2/Assign_36^save_2/Assign_37^save_2/Assign_38^save_2/Assign_39^save_2/Assign_4^save_2/Assign_5^save_2/Assign_6^save_2/Assign_7^save_2/Assign_8^save_2/Assign_9
1
save_2/restore_allNoOp^save_2/restore_shard
[
save_3/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
r
save_3/filenamePlaceholderWithDefaultsave_3/filename/input*
dtype0*
shape: *
_output_shapes
: 
i
save_3/ConstPlaceholderWithDefaultsave_3/filename*
shape: *
dtype0*
_output_shapes
: 
Ж
save_3/StringJoin/inputs_1Const*
dtype0*
_output_shapes
: *<
value3B1 B+_temp_037c5910b2194bc2a276e0eb921c7f2a/part
{
save_3/StringJoin
StringJoinsave_3/Constsave_3/StringJoin/inputs_1*
	separator *
N*
_output_shapes
: 
S
save_3/num_shardsConst*
dtype0*
_output_shapes
: *
value	B :
^
save_3/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 
Е
save_3/ShardedFilenameShardedFilenamesave_3/StringJoinsave_3/ShardedFilename/shardsave_3/num_shards*
_output_shapes
: 
К
save_3/SaveV2/tensor_namesConst*ї
value±BЃ(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
_output_shapes
:(*
dtype0
µ
save_3/SaveV2/shape_and_slicesConst*
_output_shapes
:(*
dtype0*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
…
save_3/SaveV2SaveV2save_3/ShardedFilenamesave_3/SaveV2/tensor_namessave_3/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1v/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1v/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1v/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1v/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1v/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1v/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1*6
dtypes,
*2(
Щ
save_3/control_dependencyIdentitysave_3/ShardedFilename^save_3/SaveV2*
T0*
_output_shapes
: *)
_class
loc:@save_3/ShardedFilename
£
-save_3/MergeV2Checkpoints/checkpoint_prefixesPacksave_3/ShardedFilename^save_3/control_dependency*

axis *
T0*
_output_shapes
:*
N
Г
save_3/MergeV2CheckpointsMergeV2Checkpoints-save_3/MergeV2Checkpoints/checkpoint_prefixessave_3/Const*
delete_old_dirs(
В
save_3/IdentityIdentitysave_3/Const^save_3/MergeV2Checkpoints^save_3/control_dependency*
T0*
_output_shapes
: 
Н
save_3/RestoreV2/tensor_namesConst*ї
value±BЃ(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
_output_shapes
:(*
dtype0
Є
!save_3/RestoreV2/shape_and_slicesConst*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:(
ё
save_3/RestoreV2	RestoreV2save_3/Constsave_3/RestoreV2/tensor_names!save_3/RestoreV2/shape_and_slices*6
dtypes,
*2(*ґ
_output_shapes£
†::::::::::::::::::::::::::::::::::::::::
Ґ
save_3/AssignAssignbeta1_powersave_3/RestoreV2*
use_locking(*
_output_shapes
: *
T0*
validate_shape(* 
_class
loc:@pi/dense/bias
І
save_3/Assign_1Assignbeta1_power_1save_3/RestoreV2:1*
validate_shape(*
_output_shapes
: *
T0*
use_locking(*
_class
loc:@v/dense/bias
¶
save_3/Assign_2Assignbeta2_powersave_3/RestoreV2:2*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
: 
І
save_3/Assign_3Assignbeta2_power_1save_3/RestoreV2:3*
use_locking(*
validate_shape(*
_output_shapes
: *
T0*
_class
loc:@v/dense/bias
ђ
save_3/Assign_4Assignpi/dense/biassave_3/RestoreV2:4* 
_class
loc:@pi/dense/bias*
_output_shapes
:@*
validate_shape(*
T0*
use_locking(
±
save_3/Assign_5Assignpi/dense/bias/Adamsave_3/RestoreV2:5* 
_class
loc:@pi/dense/bias*
validate_shape(*
T0*
use_locking(*
_output_shapes
:@
≥
save_3/Assign_6Assignpi/dense/bias/Adam_1save_3/RestoreV2:6*
use_locking(*
validate_shape(* 
_class
loc:@pi/dense/bias*
T0*
_output_shapes
:@
і
save_3/Assign_7Assignpi/dense/kernelsave_3/RestoreV2:7*
validate_shape(*"
_class
loc:@pi/dense/kernel*
T0*
use_locking(*
_output_shapes

:@
є
save_3/Assign_8Assignpi/dense/kernel/Adamsave_3/RestoreV2:8*
validate_shape(*
T0*
use_locking(*
_output_shapes

:@*"
_class
loc:@pi/dense/kernel
ї
save_3/Assign_9Assignpi/dense/kernel/Adam_1save_3/RestoreV2:9*
validate_shape(*
use_locking(*"
_class
loc:@pi/dense/kernel*
T0*
_output_shapes

:@
≤
save_3/Assign_10Assignpi/dense_1/biassave_3/RestoreV2:10*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0
Ј
save_3/Assign_11Assignpi/dense_1/bias/Adamsave_3/RestoreV2:11*
use_locking(*
validate_shape(*"
_class
loc:@pi/dense_1/bias*
T0*
_output_shapes
:@
є
save_3/Assign_12Assignpi/dense_1/bias/Adam_1save_3/RestoreV2:12*
use_locking(*
T0*
validate_shape(*"
_class
loc:@pi/dense_1/bias*
_output_shapes
:@
Ї
save_3/Assign_13Assignpi/dense_1/kernelsave_3/RestoreV2:13*
T0*$
_class
loc:@pi/dense_1/kernel*
use_locking(*
validate_shape(*
_output_shapes

:@@
њ
save_3/Assign_14Assignpi/dense_1/kernel/Adamsave_3/RestoreV2:14*
_output_shapes

:@@*
use_locking(*
validate_shape(*
T0*$
_class
loc:@pi/dense_1/kernel
Ѕ
save_3/Assign_15Assignpi/dense_1/kernel/Adam_1save_3/RestoreV2:15*
use_locking(*$
_class
loc:@pi/dense_1/kernel*
T0*
validate_shape(*
_output_shapes

:@@
≤
save_3/Assign_16Assignpi/dense_2/biassave_3/RestoreV2:16*
validate_shape(*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:
Ј
save_3/Assign_17Assignpi/dense_2/bias/Adamsave_3/RestoreV2:17*"
_class
loc:@pi/dense_2/bias*
T0*
_output_shapes
:*
validate_shape(*
use_locking(
є
save_3/Assign_18Assignpi/dense_2/bias/Adam_1save_3/RestoreV2:18*
use_locking(*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
_output_shapes
:*
T0
Ї
save_3/Assign_19Assignpi/dense_2/kernelsave_3/RestoreV2:19*$
_class
loc:@pi/dense_2/kernel*
use_locking(*
T0*
validate_shape(*
_output_shapes

:@
њ
save_3/Assign_20Assignpi/dense_2/kernel/Adamsave_3/RestoreV2:20*
validate_shape(*
use_locking(*
T0*$
_class
loc:@pi/dense_2/kernel*
_output_shapes

:@
Ѕ
save_3/Assign_21Assignpi/dense_2/kernel/Adam_1save_3/RestoreV2:21*
use_locking(*
_output_shapes

:@*
T0*
validate_shape(*$
_class
loc:@pi/dense_2/kernel
ђ
save_3/Assign_22Assignv/dense/biassave_3/RestoreV2:22*
_output_shapes
:@*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
use_locking(
±
save_3/Assign_23Assignv/dense/bias/Adamsave_3/RestoreV2:23*
validate_shape(*
_class
loc:@v/dense/bias*
_output_shapes
:@*
T0*
use_locking(
≥
save_3/Assign_24Assignv/dense/bias/Adam_1save_3/RestoreV2:24*
_class
loc:@v/dense/bias*
_output_shapes
:@*
T0*
validate_shape(*
use_locking(
і
save_3/Assign_25Assignv/dense/kernelsave_3/RestoreV2:25*
_output_shapes

:@*
use_locking(*
validate_shape(*!
_class
loc:@v/dense/kernel*
T0
є
save_3/Assign_26Assignv/dense/kernel/Adamsave_3/RestoreV2:26*
use_locking(*
T0*!
_class
loc:@v/dense/kernel*
validate_shape(*
_output_shapes

:@
ї
save_3/Assign_27Assignv/dense/kernel/Adam_1save_3/RestoreV2:27*
T0*
_output_shapes

:@*
use_locking(*
validate_shape(*!
_class
loc:@v/dense/kernel
∞
save_3/Assign_28Assignv/dense_1/biassave_3/RestoreV2:28*!
_class
loc:@v/dense_1/bias*
T0*
_output_shapes
:@*
use_locking(*
validate_shape(
µ
save_3/Assign_29Assignv/dense_1/bias/Adamsave_3/RestoreV2:29*
validate_shape(*!
_class
loc:@v/dense_1/bias*
T0*
_output_shapes
:@*
use_locking(
Ј
save_3/Assign_30Assignv/dense_1/bias/Adam_1save_3/RestoreV2:30*
T0*
validate_shape(*
use_locking(*
_output_shapes
:@*!
_class
loc:@v/dense_1/bias
Є
save_3/Assign_31Assignv/dense_1/kernelsave_3/RestoreV2:31*
T0*
_output_shapes

:@@*#
_class
loc:@v/dense_1/kernel*
use_locking(*
validate_shape(
љ
save_3/Assign_32Assignv/dense_1/kernel/Adamsave_3/RestoreV2:32*#
_class
loc:@v/dense_1/kernel*
_output_shapes

:@@*
T0*
use_locking(*
validate_shape(
њ
save_3/Assign_33Assignv/dense_1/kernel/Adam_1save_3/RestoreV2:33*
_output_shapes

:@@*
use_locking(*
T0*
validate_shape(*#
_class
loc:@v/dense_1/kernel
∞
save_3/Assign_34Assignv/dense_2/biassave_3/RestoreV2:34*
_output_shapes
:*!
_class
loc:@v/dense_2/bias*
validate_shape(*
T0*
use_locking(
µ
save_3/Assign_35Assignv/dense_2/bias/Adamsave_3/RestoreV2:35*
use_locking(*!
_class
loc:@v/dense_2/bias*
validate_shape(*
_output_shapes
:*
T0
Ј
save_3/Assign_36Assignv/dense_2/bias/Adam_1save_3/RestoreV2:36*
validate_shape(*
use_locking(*
T0*!
_class
loc:@v/dense_2/bias*
_output_shapes
:
Є
save_3/Assign_37Assignv/dense_2/kernelsave_3/RestoreV2:37*
_output_shapes

:@*
use_locking(*
T0*#
_class
loc:@v/dense_2/kernel*
validate_shape(
љ
save_3/Assign_38Assignv/dense_2/kernel/Adamsave_3/RestoreV2:38*
use_locking(*
validate_shape(*
T0*
_output_shapes

:@*#
_class
loc:@v/dense_2/kernel
њ
save_3/Assign_39Assignv/dense_2/kernel/Adam_1save_3/RestoreV2:39*
validate_shape(*
use_locking(*
_output_shapes

:@*
T0*#
_class
loc:@v/dense_2/kernel
И
save_3/restore_shardNoOp^save_3/Assign^save_3/Assign_1^save_3/Assign_10^save_3/Assign_11^save_3/Assign_12^save_3/Assign_13^save_3/Assign_14^save_3/Assign_15^save_3/Assign_16^save_3/Assign_17^save_3/Assign_18^save_3/Assign_19^save_3/Assign_2^save_3/Assign_20^save_3/Assign_21^save_3/Assign_22^save_3/Assign_23^save_3/Assign_24^save_3/Assign_25^save_3/Assign_26^save_3/Assign_27^save_3/Assign_28^save_3/Assign_29^save_3/Assign_3^save_3/Assign_30^save_3/Assign_31^save_3/Assign_32^save_3/Assign_33^save_3/Assign_34^save_3/Assign_35^save_3/Assign_36^save_3/Assign_37^save_3/Assign_38^save_3/Assign_39^save_3/Assign_4^save_3/Assign_5^save_3/Assign_6^save_3/Assign_7^save_3/Assign_8^save_3/Assign_9
1
save_3/restore_allNoOp^save_3/restore_shard "&B
save_3/Const:0save_3/Identity:0save_3/restore_all (5 @F8"
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
Placeholder:0€€€€€€€€€%
pi
pi/Squeeze:0	€€€€€€€€€#
v
v/Squeeze:0€€€€€€€€€tensorflow/serving/predict