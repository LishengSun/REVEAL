from context import segment_env
# from segment_env.segment_env import *
from segment_env.segment_env import *


env_spec = {
	"max_steps": 10,
	"window": 1, 
	"segment_length": 10,
	"noise": 0.0,
	"free_location": False,
	"expl_cost":0.05,
	"pred_reward": 1
}

def test_init():
	seg_env = ImgEnv(segment_length=50, max_steps=50)
	assert hasattr(seg_env, "max_steps")
	assert hasattr(seg_env, "window")
	assert hasattr(seg_env, "segment_length")
	assert hasattr(seg_env, "noise")
	assert hasattr(seg_env, "free_location")
	assert hasattr(seg_env, "expl_cost")
	assert hasattr(seg_env, "pred_reward")
	assert hasattr(seg_env, "observation_space")
	assert hasattr(seg_env, "action_space")
	assert hasattr(seg_env, "to_draw")


def test_reset():
	seg_env = ImgEnv()
	seg_env.reset()
	assert hasattr(seg_env, "curr_img")
	assert hasattr(seg_env, "target")
	assert hasattr(seg_env, "state")
	assert isinstance(seg_env.target, float)
	assert seg_env.curr_img.shape == (50,)
	assert seg_env.state.shape == (2, 50)

def test_step(action=[1,10]):
	seg_env = ImgEnv()
	seg_env.reset()
	_, _, done = seg_env.step(action)
	assert done == True

def test_step(action=[0,10]):
	seg_env = ImgEnv()
	seg_env.reset()
	_, _, done = seg_env.step(action)
	assert done == False



















