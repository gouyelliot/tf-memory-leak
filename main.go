package main

import (
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"io/ioutil"
	"log"
	"net/http"
	_ "net/http/pprof"
)

type FaceDetector struct {
	session *tf.Session
	graph   *tf.Graph

	inputOp  tf.Output
	bboxesOp tf.Output
	scoresOp tf.Output
}

func NewFaceDetector(frozenGraphPath string) (*FaceDetector, error) {
	fd := &FaceDetector{}
	model, err := ioutil.ReadFile(frozenGraphPath)
	if err != nil {
		return nil, err
	}

	fd.graph = tf.NewGraph()
	if err := fd.graph.Import(model, ""); err != nil {
		return nil, err
	}

	fd.session, err = tf.NewSession(fd.graph, nil)
	if err != nil {
		return nil, err
	}

	fd.inputOp = fd.graph.Operation("input_image").Output(0)
	fd.bboxesOp = fd.graph.Operation("bboxes").Output(0)
	fd.scoresOp = fd.graph.Operation("scores_1/GatherV2").Output(0)

	return fd, nil
}

func (fd *FaceDetector) Close() error {
	if err := fd.session.Close(); err != nil {
		return err
	}
	return nil
}

func (fd *FaceDetector) FindFaces(image []byte) ([]float32, [][]int32, error) {
	imageTensor, err := tf.NewTensor(string(image))
	if err != nil {
		return nil, nil, err
	}

	output, err := fd.session.Run(
		map[tf.Output]*tf.Tensor{
			fd.inputOp: imageTensor,
		},
		[]tf.Output{
			fd.bboxesOp,
			fd.scoresOp,
		},
		nil)

	if err != nil {
		return nil, nil, err
	}

	bboxes := output[0].Value().([][]int32)
	scores := output[1].Value().([]float32)

	return scores, bboxes, nil
}

func main() {
	log.SetFlags(log.LstdFlags | log.Lshortfile | log.Lmicroseconds)

	go func() {
		log.Println(http.ListenAndServe("0.0.0.0:8200", nil))
	}()

	fd, err := NewFaceDetector("OptimizedGraph.pb")

	if err != nil {
		panic(err)
	}
	defer fd.Close()

	for {
		img, err := ioutil.ReadFile("faces.jpg")
		if err != nil {
			log.Println("Fail to read image:", err)
		}

		scores, boxes, err := fd.FindFaces(img)

		if err != nil {
			log.Println("Fail to infer:", err)
		}

		log.Println(scores, boxes)
	}
}
