import { createRouter, createWebHistory } from 'vue-router';
import DashboardView from '../views/DashboardView.vue';
import ClassifyView from '../views/ClassifyView.vue';
import HistoryView from '../views/HistoryView.vue';
import ModelInfoView from '../views/ModelInfoView.vue';
import AboutView from '../views/AboutView.vue';

const routes = [
  {
    path: '/',
    name: 'Dashboard',
    component: DashboardView,
    meta: { title: 'Dashboard | Hệ thống AI Phân Loại Rác' }
  },
  {
    path: '/classify',
    name: 'Classify',
    component: ClassifyView,
    meta: { title: 'Phân loại rác | Hệ thống AI Phân Loại Rác' }
  },
  {
    path: '/history',
    name: 'History',
    component: HistoryView,
    meta: { title: 'Lịch sử nhận diện | Hệ thống AI Phân Loại Rác' }
  },
  {
    path: '/model',
    name: 'ModelInfo',
    component: ModelInfoView,
    meta: { title: 'Mô hình AI YOLO11 | Hệ thống AI Phân Loại Rác' }
  },
  {
    path: '/about',
    name: 'About',
    component: AboutView,
    meta: { title: 'Giới thiệu đồ án | Hệ thống AI Phân Loại Rác' }
  },
  {
    path: '/:pathMatch(.*)*',
    redirect: '/'
  }
];

const router = createRouter({
  history: createWebHistory(),
  routes,
  scrollBehavior() {
    return { top: 0 };
  }
});

router.beforeEach((to, from, next) => {
  document.title = to.meta.title || 'Hệ thống AI Phân Loại Rác';
  next();
});

export default router;
