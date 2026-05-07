package com.example.cagent.ui

import androidx.compose.runtime.Composable
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController

private object Routes {
    const val Models = "models"
    const val Chat = "chat"
}

@Composable
fun AppNav() {
    val nav = rememberNavController()
    NavHost(navController = nav, startDestination = Routes.Models) {
        composable(Routes.Models) {
            ModelsScreen(onOpenChat = { nav.navigate(Routes.Chat) })
        }
        composable(Routes.Chat) {
            ChatScreen(onOpenModels = { nav.navigate(Routes.Models) })
        }
    }
}

