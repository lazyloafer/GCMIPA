try:
    from Queue import Queue
except ImportError:
    from queue import Queue
import random

def BFS(kb, entity1, entity2):
	# kb = {entity1:[relation entity2]}
	# entity1 起点
	# entity2 终点
	res = foundPaths(kb) # res.entities = {entity_flag, prevNode, relation} initialize to {False, "", ""}
	res.markFound(entity1, None, None) #起始节点初始化为 {True, None, None}
	q = Queue()
	q.put(entity1)#起始节点入队列
	while(not q.empty()):
		curNode = q.get() # 取最后进队的，并出队
		for path in kb.getPathsFrom(curNode):
			nextEntity = path.connected_entity
			connectRelation = path.relation
			if(not res.isFound(nextEntity)):# 若 res.isFound(nextEntity)=False 即 entity_flag=False
				q.put(nextEntity) #将未选中的实体入队列
				res.markFound(nextEntity, curNode, connectRelation)
			if(nextEntity == entity2):
				entity_list, path_list = res.reconstructPath(entity1, entity2)
				return (True, entity_list, path_list)
	return (False, None, None)

def test():
	pass

class foundPaths(object):
	def __init__(self, kb):
		self.entities = {}
		# for entity, relations in kb.entities.iteritems():
		for entity, relations in kb.entities.items():
			self.entities[entity] = (False, "", "")

	def isFound(self, entity):
		return self.entities[entity][0]
			

	def markFound(self, entity, prevNode, relation):
		self.entities[entity] = (True, prevNode, relation)

	def reconstructPath(self, entity1, entity2):
		entity_list = []
		path_list = []
		curNode = entity2
		while(curNode != entity1):
			entity_list.append(curNode)

			path_list.append(self.entities[curNode][2])
			curNode = self.entities[curNode][1]
		entity_list.append(curNode)
		entity_list.reverse()
		path_list.reverse()
		return (entity_list, path_list)

	def __str__(self):
		res = ""
		for entity, status in self.entities.iteritems():
			res += entity + "[{},{},{}]".format(status[0],status[1],status[2])
		return res			
