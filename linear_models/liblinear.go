package linear_models

const (
	L2R_LR = iota
	L2R_L2LOSS_SVC_DUAL
	L2R_L2LOSS_SVC
	L2R_L1LOSS_SVC_DUAL
	MCSVM_CS
	L1R_L2LOSS_SVC
	L1R_LR
	L2R_LR_DUAL
)

type Parameter struct {
	solver_type  int
	eps          float64
	C            float64
	nr_weight    int
	weight_label []int
	weight       []float64
	p            float64
}

type Problem struct {
	l    int
	n    int
	y    []float64
	x    []*FeatureNode
	bias float64
}

type Model struct {
	param      Parameter
	nr_class   int
	nr_feature int
	w          []float64
	label      []int
	bias       float64
}

type FeatureNode struct {
	index int
	value float64
}

func NewParameter(solver_type int, C float64, eps float64) *Parameter {
	param := Parameter{}
	param.solver_type = solver_type
	param.eps = eps
	param.C = C
	param.nr_weight = 0
	param.weight_label = make([]int, 0)
	param.weight = nil

	return &param
}

func NewProblem(X [][]float64, y []float64, bias float64) *Problem {
	prob := Problem{}
	prob.l = len(X)
	prob.n = len(X[0]) + 1

	prob.x = convert_features(X, bias)
	c_y := make([]float64, len(y))
	for i := 0; i < len(y); i++ {
		c_y[i] = y[i]
	}
	prob.y = c_y
	prob.bias = -1

	return &prob
}

// TODO:
func Train(prob *Problem, param *Parameter) *Model {

	return &Model{}
}

// TODO:
func Export(model *Model, filePath string) error {
	return nil
}

// TODO:
func Load(model *Model, filePath string) error {
	return nil
}

// TODO:
func Predict(model *Model, x []float64) float64 {
	return 0
}

func convert_vector(x []float64, bias float64) *FeatureNode {
	n_ele := 0
	for i := 0; i < len(x); i++ {
		if x[i] > 0 {
			n_ele++
		}
	}
	n_ele += 2

	c_x := make([]FeatureNode, n_ele)
	j := 0
	for i := 0; i < len(x); i++ {
		if x[i] > 0 {
			c_x[j].index = i + 1
			c_x[j].value = x[i]
			j++
		}
	}
	if bias > 0 {
		c_x[j].index = 0
		c_x[j].value = 0
		j++
	}
	c_x[j].index = -1
	return &c_x[0]
}
func convert_features(X [][]float64, bias float64) []*FeatureNode {
	n_samples := len(X)
	n_elements := 0

	for i := 0; i < n_samples; i++ {
		for j := 0; j < len(X[i]); j++ {
			if X[i][j] != 0.0 {
				n_elements++
			}
			n_elements++ //for bias
		}
	}

	x_space := make([]FeatureNode, n_elements+n_samples)

	cursor := 0
	x := make([]*FeatureNode, n_samples)

	for i := 0; i < n_samples; i++ {
		x[i] = &x_space[cursor]

		for j := 0; j < len(X[i]); j++ {
			if X[i][j] != 0.0 {
				x_space[cursor].index = j + 1
				x_space[cursor].value = X[i][j]
				cursor++
			}
			if bias > 0 {
				x_space[cursor].index = 0
				x_space[cursor].value = bias
				cursor++
			}
		}
		x_space[cursor].index = -1
		cursor++
	}

	return x
}
