(define (problem game-3)
  (:domain game)
  (:objects
    l0 l1 - coord
  )
  (:init
    (on l0 l0 Y)
    (on l0 l1 Y)
    (on l1 l0 Y)
    (on l1 l1 Y)
    (next l0 l1)
    (max l1)
    (min l0)
  )
  (:htn
    :tasks (and
      (play)
    )
  )
  (:goal (and
    (on l0 l0 Y)
    (on l0 l1 Y)
    (on l1 l0 N)
    (on l1 l1 Y)
  ))
)
