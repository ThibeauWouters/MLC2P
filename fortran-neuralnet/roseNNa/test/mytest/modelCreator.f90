module model
    !---------------
    ! adding any number of layers to our neural network
    !----------------


    ! ===============================================================
    ! USE filereader !<loading in weights, biases
    USE activation_functions !<getting activation functions
    USE model_layers
    USE reader
    USE iso_c_binding
    ! ===============================================================


    IMPLICIT NONE
    contains
    !===============================================THIS BLOCK COMES FROM FYPP OUTPUT "openNP.fpp"=======================================================

        SUBROUTINE use_model(        i0,         o0) bind(c,name="use_model")
            IMPLICIT NONE
            !INPUTS CORRESPONDING TO C
            REAL (c_double), INTENT(INOUT), DIMENSION(        1, 3) :: i0
            !===========================

            !INPUTS CORRESPONDING TO INTERMEDIARY PROCESSING
            !================================================

            !OUTPUTS CORRESPONDING TO C
            REAL (c_double), INTENT(OUT), DIMENSION(        1, 1) :: o0
            !===========================

            REAL (c_double), ALLOCATABLE, DIMENSION(:,:) :: input

            REAL :: T1, T2

            ALLOCATE(input(        1, 3))

            input = i0


            CALL CPU_TIME(T1)

            !========Gemm Layer============
            CALL linear_layer(input, linLayers(1),0)

            input = sigmoid2d(input)

            !========Gemm Layer============
            CALL linear_layer(input, linLayers(2),0)

            input = sigmoid2d(input)

            !========Gemm Layer============
            CALL linear_layer(input, linLayers(3),0)

            call CPU_TIME(T2)
            o0 = input
        end SUBROUTINE
    !===================================================================================================================================================


END module model

!=======================================================================================
