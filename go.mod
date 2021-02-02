module github.com/gouyelliot/tf-memory-leak

go 1.15

require github.com/tensorflow/tensorflow v2.4.1+incompatible

// Offical repo doesn't include the proto files
replace github.com/tensorflow/tensorflow => github.com/onfocusio/tensorflow-go v1.0.1-0.20210202105807-3e3be48522a7
