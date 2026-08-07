package it.unibo.examples.cooperative

import it.unibo.examples.cooperative.BoundedWorldEnvironment.State
import it.unibo.model.core.abstractions.{AI, Enumerable, MultiAgentEnvironment}
import it.unibo.model.core.learning.Learner

import scala.util.Random

class BoundedWorldEnvironment(using random: Random)(
    agents: Int,
    boundSize: Int,
    obstacles: Set[(Int, Int)] = Set.empty,
    obstacleHitPenalty: Double = -5.0,
    obstacleNearPenalty: Double = -0.15,
    fixedInitial: Option[State] = None
) extends MultiAgentEnvironment[BoundedWorldEnvironment.State, BoundedWorldEnvironment.Action]:
  import BoundedWorldEnvironment.*

  var state: State = fixedInitial.getOrElse(generatePosition)

  private def isNearObstacle(position: (Int, Int)): Boolean =
    obstacles.exists { obstacle =>
      val dx = math.abs(position._1 - obstacle._1)
      val dy = math.abs(position._2 - obstacle._2)

      dx + dy == 1
    }

  def act(actions: Seq[Action]): Seq[Double] =
    val prevDistance = computeAlignmentReward(alignmentDistance(state))
    val transitions =
      state.zip(actions).map { case (currentPosition, action) =>
        val proposedPosition =
          action.updatePosition(currentPosition, boundSize)

        if obstacles.contains(proposedPosition) then
          (currentPosition, true)
        else
          (proposedPosition, false)
      }

    state = transitions.map(_._1)
    val alignmentReward =
      computeAlignmentReward(prevDistance)

    transitions.zip(state).map {
      case ((_, true), _) =>
        alignmentReward + obstacleHitPenalty

      case ((_, false), position) if isNearObstacle(position) =>
        alignmentReward + obstacleNearPenalty

      case _ =>
        alignmentReward
    }

  override def reset(): Unit = state = fixedInitial.getOrElse(generatePosition)

  private def alignmentDistance(state: State): Int =
    val rows = state.map(_._2)
    rows.max - rows.min
  private def computeAlignmentReward(previousDistance: Double): Double =
      val newDistance = alignmentDistance(state)
      val alignmentReward =
        if newDistance == 0 then
          5.0
        else if newDistance < previousDistance then
          1.0
        else if newDistance > previousDistance then
          -1.0
        else
          -0.05
      alignmentReward    

  private def generatePosition: State =
    val half = boundSize / 2

    (0 until agents).map { agentIndex =>
      val rowStart =
        (agentIndex % 2) * half

      val freePositions =
        for
          x <- 0 until boundSize
          y <- rowStart until (rowStart + half).min(boundSize)
          position = (x, y)
          if !obstacles.contains(position)
        yield position
      freePositions(random.nextInt(freePositions.size))
    }.toList

enum MovementAction derives Enumerable:
  case Up, Down, Right, Left, NoOp
  def updatePosition(position: (Int, Int), bound: Int): (Int, Int) =
    val (x, y) = position
//    def pacmanEffect(coordinate: Int): Int = if coordinate < 0 then bound - 1 else coordinate
//
//      this match
//            case Up => (x, (y + 1) % bound)
//            case Down => (x, pacmanEffect(y - 1))
//            case Left => (pacmanEffect(x - 1), y)
//            case Right => ((x + 1) % bound, y)
//            case NoOp => position
    this match
        case Up => (x, (y - 1).max(0))
        case Down => (x, (y + 1).min(bound - 1))
        case Left => ((x - 1).max(0), y)
        case Right => ((x + 1).min(bound - 1), y)
        case NoOp => position

object BoundedWorldEnvironment:
  type State = List[(Int, Int)]
  type Action = MovementAction

  case class RelativeState(rowDiff: Int, colDiff: Int, obstacleDiffX: Int, obstacleDiffY: Int, obstacleVisible: Boolean)

  def toRelative(state: State, agentIndex: Int, boundSize: Int, obstacles: Set[(Int, Int)], visionRange: Int): RelativeState =
    val (myRow, myCol) = state(agentIndex)
    val otherIndex = if agentIndex == 0 then 1 else 0
    val (otherRow, otherCol) = state(otherIndex)
    val rowDiff = otherRow - myRow
    val colDiff = otherCol - myCol
    val visibleObstacles =
      obstacles.filter { case (x, y) =>
        math.abs(x - myRow) <= visionRange &&
          math.abs(y - myCol) <= visionRange
      }

    val nearest =
      visibleObstacles.minByOption { case (x, y) =>
        math.abs(x - myRow) + math.abs(y - myCol)
      }

    nearest match
      case Some((ox, oy)) =>
        RelativeState(
          rowDiff,
          colDiff,
          ox - myRow,
          oy - myCol,
          true
        )

      case None =>
        RelativeState(
          rowDiff,
          colDiff,
          0,
          0,
          false
        )

  def fromRelative(rs: RelativeState, myPos: (Int, Int), boundSize: Int): (Int, Int) =
    ((myPos._1 + rs.rowDiff) % boundSize, (myPos._2 + rs.colDiff) % boundSize)

  class RelativeStateAgent(
                            qAgent: AI.Agent[RelativeState, Action] & Learner[RelativeState, Action],
                            agentIndex: Int,
                            boundSize: Int,
                            obstacles: Set[(Int, Int)],
                            visionRange: Int
  ) extends AI.Agent[State, Action], Learner[State, Action]:
    override def optimal: State => Action = s => qAgent.optimal(toRelative(s, agentIndex, boundSize, obstacles, visionRange))
    override def behavioural: State => Action = s => qAgent.behavioural(toRelative(s, agentIndex, boundSize, obstacles, visionRange))
    override def improve(state: State, action: Action, reward: Double, nextState: State, done: Boolean): Unit =
      qAgent.improve(toRelative(state, agentIndex, boundSize, obstacles, visionRange), action, reward, toRelative(nextState, agentIndex, boundSize, obstacles, visionRange), done)
    override def reset(): Unit = qAgent.reset()
