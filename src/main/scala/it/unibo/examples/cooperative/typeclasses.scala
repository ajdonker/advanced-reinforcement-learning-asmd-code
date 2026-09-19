package it.unibo.examples.cooperative

import BoundedWorldEnvironment.RelativeState
import it.unibo.model.core.network.NeuralNetworkEncoding

class StateEncoding(agents: Int, maxBound: Float) extends NeuralNetworkEncoding[List[(Int, Int)]]:
  private val stateSpaceSize = 2
  override def elements: Int = agents * stateSpaceSize
  override def toSeq(elem: List[(Int, Int)]): Seq[Double] = elem.flatMap:
    case (l, r) =>
      List(l / maxBound, r / maxBound)

class RelativeStateEncoding(boundSize: Int, visionRange: Int) extends NeuralNetworkEncoding[RelativeState]:
  private val noObstacle = visionRange + 1

  override def elements: Int = 6

  override def toSeq(elem: RelativeState): Seq[Double] =
    List(
      elem.rowDiff.toDouble / boundSize,
      elem.colDiff.toDouble / boundSize,
      elem.obstacleUp.toDouble / noObstacle,
      elem.obstacleDown.toDouble / noObstacle,
      elem.obstacleLeft.toDouble / noObstacle,
      elem.obstacleRight.toDouble / noObstacle
    )
